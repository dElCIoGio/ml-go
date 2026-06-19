package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"ml/matrix"
	"ml/types"
)

type ModelProgram struct {
	Vars []*Tensor
}

type ModelContext struct {
	NumberOfVars int

	Input         *Tensor // = x
	Output        *Tensor // = prediction
	DesiredOutput *Tensor // = target/label
	Cost          *Tensor // = loss

	ForwardProgram *ModelProgram
	CostProgram    *ModelProgram
}

type ModelTrainingDesc struct {
	TrainImages *matrix.Matrix[float64]
	TestImages  *matrix.Matrix[float64]
	TrainLabels *matrix.Matrix[float64]
	TestLabels  *matrix.Matrix[float64]

	Epochs       int
	BatchSize    int
	LearningRate float32
}

func ModelProgramCreate(outTensor *Tensor) *ModelProgram {
	visited := map[*Tensor]bool{}
	var vars []*Tensor

	var visit func(t *Tensor)

	visit = func(t *Tensor) {
		if t == nil {
			return
		}

		if visited[t] {
			return
		}

		visited[t] = true

		for _, input := range t.Inputs {
			visit(input)
		}

		vars = append(vars, t)
	}

	visit(outTensor)

	return &ModelProgram{
		Vars: vars,
	}
}
func (prog *ModelProgram) Compute() {
	for i := 0; i < len(prog.Vars); i++ {
		curr := prog.Vars[i]

		switch curr.Operation {
		case types.OpCreate:
			continue

		case types.OneInputOp, types.TwoInputOp:
			continue

		case types.OpAdd:
			a := curr.Inputs[0]
			b := curr.Inputs[1]

			val, err := a.Data.Add(b.Data)
			if err != nil {
				panic(err)
			}

			curr.Data = val

		case types.OpSub:
			a := curr.Inputs[0]
			b := curr.Inputs[1]

			val, err := a.Data.Sub(b.Data)
			if err != nil {
				panic(err)
			}

			curr.Data = val

		case types.OpMatMul:
			a := curr.Inputs[0]
			b := curr.Inputs[1]

			val, err := a.Data.MatMul(b.Data)
			if err != nil {
				panic(err)
			}

			curr.Data = val

		case types.OpNeg:
			a := curr.Inputs[0]
			curr.Data = mapMatrix(a.Data, func(x float64) float64 {
				return -x
			})

		case types.OpExp:
			a := curr.Inputs[0]
			curr.Data = mapMatrix(a.Data, func(x float64) float64 {
				return math.Exp(x)
			})

		case types.OpLog:
			a := curr.Inputs[0]
			curr.Data = mapMatrix(a.Data, func(x float64) float64 {
				return math.Log(x)
			})

		case types.OpMul:
			a := curr.Inputs[0]
			b := curr.Inputs[1]

			curr.Data = elementwise(a.Data, b.Data, func(x, y float64) float64 {
				return x * y
			})

		case types.OpDiv:
			a := curr.Inputs[0]
			b := curr.Inputs[1]
			curr.Data = elementwiseBroadcast(a.Data, b.Data, func(x, y float64) float64 {
				if y == 0 {
					panic("division by zero")
				}

				return x / y
			})

		case types.OpMaximum:
			a := curr.Inputs[0]
			b := curr.Inputs[1]

			curr.Data = elementwise(a.Data, b.Data, func(x, y float64) float64 {
				if x > y {
					return x
				}
				return y
			})

		case types.OpSum:
			a := curr.Inputs[0]

			sum := 0.0
			for row := 0; row < a.Data.Rows; row++ {
				for col := 0; col < a.Data.Cols; col++ {
					sum += a.Data.At(row, col)
				}
			}

			m := matrix.NewEmptyMatrix[float64](1, 1)
			m.Set(0, 0, sum)
			curr.Data = &m

		case types.OpMean:
			a := curr.Inputs[0]

			sum := 0.0
			count := float64(a.Data.Rows * a.Data.Cols)

			for row := 0; row < a.Data.Rows; row++ {
				for col := 0; col < a.Data.Cols; col++ {
					sum += a.Data.At(row, col)
				}
			}

			m := matrix.NewEmptyMatrix[float64](1, 1)
			m.Set(0, 0, sum/count)
			curr.Data = &m

		case types.OpTranspose:
			a := curr.Inputs[0]
			curr.Data = transposeMatrix(a.Data)

		default:
			panic("unhandled operation")
		}
	}
}

func (prog *ModelProgram) ComputeGrads() {
	if len(prog.Vars) == 0 {
		return
	}

	for _, curr := range prog.Vars {
		if curr.HasFlag(types.RequiresGradFlag) && curr.Grad != nil {
			curr.Grad.Clear()
		}
	}

	lastIndex := len(prog.Vars) - 1
	last := prog.Vars[lastIndex]

	if last.Grad == nil {
		grad := matrix.NewEmptyMatrix[float64](last.Data.Rows, last.Data.Cols)
		last.Grad = &grad
	}

	last.Grad.Fill(1)

	for i := lastIndex; i >= 0; i-- {
		curr := prog.Vars[i]

		if curr.Backward != nil {
			curr.Backward()
		}
	}
}

func ModelCreate() *ModelContext {

	model := ModelContext{}

	return &model
}

func ModelCompile(ctx *ModelContext) {

	if ctx.Output != nil {
		ctx.ForwardProgram = ModelProgramCreate(ctx.Output)
	}

	if ctx.Cost != nil {
		ctx.CostProgram = ModelProgramCreate(ctx.Cost)
	}

}
func ModelFeedForward(ctx *ModelContext) {
	ctx.ForwardProgram.Compute()
}
func ModelTrain(ctx *ModelContext, desc *ModelTrainingDesc) {
	numExamples := desc.TrainImages.Rows
	numBatches := numExamples / desc.BatchSize

	trainingOrder := make([]int, numExamples)
	for i := 0; i < numExamples; i++ {
		trainingOrder[i] = i
	}

	for epoch := 0; epoch < desc.Epochs; epoch++ {
		rand.Shuffle(numExamples, func(i, j int) {
			trainingOrder[i], trainingOrder[j] = trainingOrder[j], trainingOrder[i]
		})

		for batch := 0; batch < numBatches; batch++ {
			ctx.CostProgram.ZeroGrads(true)

			avgCost := 0.0

			for i := 0; i < desc.BatchSize; i++ {

				ctx.CostProgram.ZeroGrads(false)
				ctx.CostProgram.Compute()
				ctx.CostProgram.Backward()

				avgCost += ctx.Cost.Data.At(0, 0)
			}

			avgCost /= float64(desc.BatchSize)

			UpdateParameters(ctx.CostProgram, float64(desc.LearningRate), desc.BatchSize)

			fmt.Printf(
				"Epoch %d/%d, Batch %d/%d, Average Cost: %.4f\r",
				epoch+1,
				desc.Epochs,
				batch+1,
				numBatches,
				avgCost,
			)
		}

		fmt.Println()
	}
}

func (prog *ModelProgram) Backward() {
	if len(prog.Vars) == 0 {
		return
	}

	last := prog.Vars[len(prog.Vars)-1]

	if last.Grad == nil {
		g := matrix.NewEmptyMatrix[float64](last.Data.Rows, last.Data.Cols)
		last.Grad = &g
	}

	last.Grad.Fill(1)

	for i := len(prog.Vars) - 1; i >= 0; i-- {
		curr := prog.Vars[i]

		if curr.Backward != nil {
			curr.Backward()
		}
	}
}

func (prog *ModelProgram) ZeroGrads(includeParams bool) {
	for _, t := range prog.Vars {
		if t.Grad == nil {
			continue
		}

		if !includeParams && t.HasFlag(types.ParameterFlag) {
			continue
		}

		t.Grad.Clear()
	}
}

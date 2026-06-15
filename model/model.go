package model

import (
	"math"
	"ml/matrix"
	"ml/tensor"
	"ml/types"
)

type ModelProgram struct {
	Vars []*tensor.Tensor
}

type ModelContext struct {
	NumberOfVars int

	Input         *tensor.Tensor // = x
	Output        *tensor.Tensor // = prediction
	DesiredOutput *tensor.Tensor // = target/label
	Cost          *tensor.Tensor // = loss

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

func ModelProgramCreate(outTensor *tensor.Tensor) *ModelProgram {
	visited := map[*tensor.Tensor]bool{}
	var vars []*tensor.Tensor

	var visit func(t *tensor.Tensor)

	visit = func(t *tensor.Tensor) {
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

			curr.Data = elementwise(a.Data, b.Data, func(x, y float64) float64 {
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
	prog.Vars[lastIndex].Grad.Fill(float64(1))

	for i := lastIndex; i >= 0; i-- {
		curr := prog.Vars[i]

		if curr.Backward != nil {
			curr.Backward()
		}
	}
}

// func ModelCreate() *ModelContext                            {}
func ModelCompile(ctx *ModelContext)                        {}
func ModelFeedForward(ctx *ModelContext)                    {}
func ModelTrain(ctx *ModelContext, desc *ModelTrainingDesc) {}

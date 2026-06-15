package model

import (
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

		var a, b *tensor.Tensor

		numInputs := tensor.TensorNumInputs(curr.Operation)
		if numInputs >= 1 {
			a = curr.Inputs[0]
		}

		if numInputs >= 2 {
			b = curr.Inputs[1]
		}
		switch curr.Operation {
		case types.OpCreate:
			break

		case types.OneInputOp:
			break

		case types.TwoInputOp:
			break

		case types.OpAdd:
			curr.Data, _ = a.Data.Add(b.Data)
			break
		case types.OpSub:
			curr.Data, _ = a.Data.Sub(b.Data)
			break
		case types.OpMatMul:
			curr.Data, _ = a.Data.Mul(b.Data)
			break
		default:
			panic("unhandled default case")

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
	prog.Vars[lastIndex].
		Grad.Fill(float64(1))

	for i := lastIndex; i >= 0; i-- {
		curr := prog.Vars[i]

		var a, b *tensor.Tensor

		numInputs := tensor.TensorNumInputs(curr.Operation)
		if numInputs >= 1 {
			a = curr.Inputs[0]
		}

		if numInputs >= 2 {
			b = curr.Inputs[1]
		}

		if numInputs == 1 &&
			!a.HasFlag(types.RequiresGradFlag) {
			continue
		}

		if numInputs == 2 &&
			(a.HasFlag(types.ParameterFlag) &&
				b.HasFlag(types.RequiresGradFlag)) {
			continue
		}

		switch curr.Operation {
		case types.OpCreate:
			continue

		case types.OneInputOp:
		case types.TwoInputOp:
			continue

		case types.OpAdd:
			if a.HasFlag(types.RequiresGradFlag) {
				a.Grad, _ = a.Grad.Add(curr.Grad)
			}

			if b.HasFlag(types.RequiresGradFlag) {
				b.Grad, _ = b.Grad.Add(curr.Grad)
			}
			break
		case types.OpSub:
			if a.HasFlag(types.RequiresGradFlag) {
				a.Grad, _ = a.Grad.Add(curr.Grad)
			}
			if b.HasFlag(types.RequiresGradFlag) {
				b.Grad, _ = b.Grad.Add(curr.Grad)
			}

			break
		case types.OpMatMul:
			if a.HasFlag(types.RequiresGradFlag) {
				gradA, _ := curr.Grad.Mul(
					b.Data,
					matrix.MatMulOptions{TransposeB: true})
				a.Grad, _ = a.Grad.Add(gradA)
			}
			if b.HasFlag(types.RequiresGradFlag) {
				gradB, _ := a.Data.Mul(
					curr.Grad,
					matrix.MatMulOptions{TransposeA: true})
				b.Grad, _ = b.Grad.Add(gradB)
			}
			break
		default:
			panic("unhandled default case")

		}

	}
}

// func ModelCreate() *ModelContext                            {}
func ModelCompile(ctx *ModelContext)                        {}
func ModelFeedForward(ctx *ModelContext)                    {}
func ModelTrain(ctx *ModelContext, desc *ModelTrainingDesc) {}

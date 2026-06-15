package model

import (
	"ml/matrix"
	"ml/tensor"
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
func ModelProgramCompute(prog *ModelProgram)      {}
func ModelProgramComputeGrads(prog *ModelProgram) {}

// func ModelCreate() *ModelContext                            {}
func ModelCompile(ctx *ModelContext)                        {}
func ModelFeedForward(ctx *ModelContext)                    {}
func ModelTrain(ctx *ModelContext, desc *ModelTrainingDesc) {}

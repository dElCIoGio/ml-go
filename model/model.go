package model

import (
	"ml/matrix"
	"ml/tensor"
)

type Program struct {
	Vars []*tensor.Tensor
}

type Context struct {
	NumberOfVars int

	Input         *tensor.Tensor // = x
	Output        *tensor.Tensor // = prediction
	DesiredOutput *tensor.Tensor // = target/label
	Cost          *tensor.Tensor // = loss

	ForwardProgram *Program
	CostProgram    *Program
}

type TrainingDesc struct {
	TrainImages *matrix.Matrix[float64]
	TestImages  *matrix.Matrix[float64]
	TrainLabels *matrix.Matrix[float64]
	TestLabels  *matrix.Matrix[float64]

	Epochs       int
	BatchSize    int
	LearningRate float32
}

func ModelProgramCreate(ctx *Context, outTensor *tensor.Tensor) Program {}
func ModelProgramCompute(prog *Program)                                 {}
func ModelProgramComputeGrads(prog *Program)                            {}
func ModelCreate() *Context                                             {}
func ModelCompile(ctx *Context)                                         {}
func ModelFeedForward(ctx *Context)                                     {}
func ModelTrain(ctx *Context, desc *TrainingDesc)                       {}

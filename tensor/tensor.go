package tensor

import (
	"ml/matrix"
	"ml/types"
)

type OpParams struct {
	Axis     int
	KeepDims bool
	Power    float64
}

type TensorOpOptions func(*OpParams)

func WithAxis(axis int) TensorOpOptions {
	return func(op *OpParams) {
		op.Axis = axis
	}
}

func WithKeepDims(keepDims bool) TensorOpOptions {
	return func(op *OpParams) {
		op.KeepDims = keepDims
	}
}

func WithPower(power float64) TensorOpOptions {
	return func(op *OpParams) {
		op.Power = power
	}
}

type Tensor struct {
	Data *matrix.Matrix[float64]
	Grad *matrix.Matrix[float64]

	Operation types.Op
	Params    OpParams
	Inputs    []*Tensor

	Backward func()

	Flags types.TensorFlag
}

func (t *Tensor) HasFlag(flag types.TensorFlag) bool {
	return t.Flags&flag == flag
}

func (t *Tensor) AddFlag(flag types.TensorFlag) {
	t.Flags |= flag
}

type TParams struct {
	Flags     types.TensorFlag
	Operation types.Op
}

type TensorOptions func(*TParams)

func Flags(flags types.TensorFlag) TensorOptions {
	return func(op *TParams) {
		op.Flags |= flags

	}
}

func Operation(op types.Op) TensorOptions {
	return func(p *TParams) {
		p.Operation = op
	}
}

func NewTensor(data *matrix.Matrix[float64], options ...TensorOptions) *Tensor {

	params := TParams{}
	for _, option := range options {
		option(&params)
	}

	output := Tensor{
		Data:      data,
		Operation: types.OpCreate,
		Inputs:    []*Tensor{},
	}

	if params.Flags&types.RequiresGradFlag != 0 {
		grad := matrix.NewEmptyMatrix[float64](data.Rows, data.Cols)
		output.Grad = &grad
	}

	if params.Operation != types.OpCreate {
		output.Operation = params.Operation
	}

	return &output
}

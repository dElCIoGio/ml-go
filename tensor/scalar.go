package tensor

import (
	"ml/matrix"
	"ml/types"
)

func Scalar(val float64) *Tensor {

	return &Tensor{
		Data:      NewScalar[float64](val),
		Grad:      nil,
		Operation: 0,
		Inputs:    nil,
		Backward:  nil,
		Flags:     0,
	}
}

func NewScalar[T types.Number](val T) *matrix.Matrix[T] {
	m := matrix.NewEmptyMatrix[T](1, 1)
	m.Set(0, 0, val)
	return &m
}

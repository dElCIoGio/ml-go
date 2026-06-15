package matrix

import (
	"ml/tensor"
	"ml/types"
)

func Scalar(val float64) *tensor.Tensor {

	return &tensor.Tensor{
		Data:      NewScalar[float64](val),
		Grad:      nil,
		Operation: 0,
		Inputs:    nil,
		Backward:  nil,
		Flags:     0,
	}
}

func NewScalar[T types.Number](val T) *Matrix[T] {
	matrix := NewEmptyMatrix[T](1, 1)
	matrix.Set(0, 0, val)
	return &matrix
}

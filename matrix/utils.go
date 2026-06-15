package matrix

import (
	"ml/types"
)

func IsSameShape[T types.Number](a, b Matrix[T]) bool {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return false
	}
	return true
}

func AssertIsSameShape[T types.Number](a, b Matrix[T]) {
	if !IsSameShape(a, b) {
		panic("The matrices must be of the same shape")
	}
}

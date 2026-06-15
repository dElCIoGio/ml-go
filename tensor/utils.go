package tensor

import (
	"ml/matrix"
	"ml/types"
)

func _unaryOp(input *Tensor, rows, cols int, flags types.TensorFlag, operation types.Op) *Tensor {
	if input.HasFlag(types.RequiresGrad) {
		flags |= types.RequiresGrad
	}

	data := matrix.NewEmptyMatrix[float64](rows, cols)
	out := NewTensor(
		&data,
		Flags(flags),
		Operation(operation))

	out.Inputs = append(out.Inputs, input)

	return out
}

func _binaryOp(a, b *Tensor, rows, cols int, flags types.TensorFlag, operation types.Op) *Tensor {
	if a.HasFlag(types.RequiresGrad) || b.HasFlag(types.RequiresGrad) {
		flags |= types.RequiresGrad
	}

	data := matrix.NewEmptyMatrix[float64](rows, cols)
	out := NewTensor(
		&data,
		Flags(flags),
		Operation(operation))

	out.Inputs = append(out.Inputs, a)
	out.Inputs = append(out.Inputs, b)

	return out
}

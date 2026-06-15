package tensor

import (
	"ml/matrix"
	"ml/types"
)

func _unaryOp(input *Tensor, rows, cols int, flags types.TensorFlag, operation types.Op) *Tensor {
	if input.HasFlag(types.RequiresGradFlag) {
		flags |= types.RequiresGradFlag
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
	if a.HasFlag(types.RequiresGradFlag) || b.HasFlag(types.RequiresGradFlag) {
		flags |= types.RequiresGradFlag
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

func TensorNumInputs(operation types.Op) int {

	inputs := 0

	if operation > types.OneInputOp {
		inputs = 1
	}
	if operation > types.TwoInputOp {
		inputs = 2
	}

	return inputs
}

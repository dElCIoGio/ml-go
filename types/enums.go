package types

type TensorFlag uint32

const (
	None TensorFlag = 0

	RequiresGradFlag TensorFlag = 1 << (iota - 1)
	ParameterFlag
	InputFlag
	OutputFlag
	DesiredOutputFlag
	CostFlag
)

type Op int

const (
	OpCreate Op = iota

	OneInputOp

	OpNeg
	OpTranspose

	OpExp
	OpLog
	OpPow

	OpSum
	OpMean
	OpMax // reduction max over a tensor

	TwoInputOp

	OpAdd
	OpSub
	OpMatMul // A @ B (matrix multiplication)
	OpMul    // A * B (element-wise)
	OpDiv    // element-wise

	OpMaximum // element-wise max between two tensors

)

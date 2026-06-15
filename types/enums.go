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
	OpMax

	TwoInputOp

	OpAdd
	OpSub
	OpMatMul
	OpDiv
)

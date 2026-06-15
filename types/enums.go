package types

type TensorFlag uint32

const (
	None TensorFlag = 0

	RequiresGrad TensorFlag = 1 << (iota - 1)
	Parameter
	Input
	Output
	DesiredOutput
	Cost
)

type Op int

const (
	OpCreate Op = iota

	_OneInputOp

	OpNeg
	OpTranspose

	OpExp
	OpLog
	OpPow

	OpSum
	OpMean
	OpMax

	_TwoInputOp

	OpAdd
	OpSub
	OpMul
	OpDiv
	OpMatMul
)

func MVNumInputs(operation Op) int {

	inputs := 0

	if operation > _OneInputOp {
		inputs = 1
	}
	if operation > _TwoInputOp {
		inputs = 2
	}

	return inputs
}

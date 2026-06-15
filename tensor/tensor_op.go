package tensor

import (
	"ml/matrix"
	"ml/types"
)

func Add(a, b *Tensor) (*Tensor, error) {

	out := _binaryOp(a, b, a.Data.Cols, a.Data.Rows, types.None, types.OpAdd)

	val, err := a.Data.Add(b.Data)
	if err != nil {
		return nil, err
	}

	out.Data = val

	out.Backward = func() {

		if a.HasFlag(types.RequiresGradFlag) {
			a.Grad, _ = a.Grad.Add(out.Grad)
		}

		if b.HasFlag(types.RequiresGradFlag) {
			b.Grad, _ = b.Grad.Add(out.Grad)
		}
	}

	return out, nil
}

func Sub(a, b *Tensor) (*Tensor, error) {

	out := _binaryOp(a, b, a.Data.Cols, a.Data.Rows, types.None, types.OpSub)

	val, err := a.Data.Sub(b.Data)
	if err != nil {
		return nil, err
	}

	out.Data = val

	out.Backward = func() {

		if a.HasFlag(types.RequiresGradFlag) {
			a.Grad, _ = a.Grad.Sub(out.Grad)
		}

		if b.HasFlag(types.RequiresGradFlag) {
			b.Grad, _ = b.Grad.Sub(out.Grad)
		}
	}

	return out, nil
}

func Mul(a, b *Tensor) (*Tensor, error) {

	val, err := a.Data.Mul(b.Data)
	if err != nil {
		return nil, err
	}

	out := &Tensor{
		Data:      val,
		Operation: types.OpMatMul,
		Inputs:    []*Tensor{a, b},
	}

	grad := matrix.NewEmptyMatrix[float64](a.Data.Rows, a.Data.Cols)
	out.Grad = &grad

	out.Backward = func() {
		val, err := b.Data.Mul(out.Grad)
		if err != nil {
			panic(err)
		}
		a.Grad, _ = a.Grad.Add(val)

		val, err = a.Data.Mul(out.Grad)
		if err != nil {
			panic(err)
		}
		b.Grad, _ = b.Grad.Add(val)
	}

	return out, nil
}

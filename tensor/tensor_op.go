package tensor

import (
	"fmt"
	"math"
	"ml/matrix"
	"ml/types"
)

func Add(a, b *Tensor) *Tensor {

	out := _binaryOp(a, b, a.Data.Rows, a.Data.Cols, types.None, types.OpAdd)

	val, err := a.Data.Add(b.Data)
	if err != nil {
		panic(fmt.Errorf(err.Error()))

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

	return out
}

func Sub(a, b *Tensor) *Tensor {

	out := _binaryOp(
		a, b,
		a.Data.Cols, a.Data.Rows,
		types.None, types.OpSub)

	val, err := a.Data.Sub(b.Data)
	if err != nil {
		panic(err)
	}

	out.Data = val

	out.Backward = func() {

		if a.HasFlag(types.RequiresGradFlag) {
			a.Grad, _ = a.Grad.Add(out.Grad)
		}

		if b.HasFlag(types.RequiresGradFlag) {
			b.Grad, _ = b.Grad.Sub(out.Grad)
		}
	}

	return out
}

func MatMul(a, b *Tensor) *Tensor {

	val, err := a.Data.MatMul(b.Data)
	if err != nil {
		panic(err)
	}

	out := _binaryOp(a, b, a.Data.Rows, a.Data.Cols, types.None, types.OpMatMul)
	out.Data = val

	out.Backward = func() {
		if a.HasFlag(types.RequiresGradFlag) {
			bT := b.Data.Transpose()

			gradA, err := out.Grad.MatMul(&bT)
			if err != nil {
				panic(err)
			}

			a.Grad, _ = a.Grad.Add(gradA)
		}

		if b.HasFlag(types.RequiresGradFlag) {

			aT := a.Data.Transpose()

			gradB, err := aT.MatMul(out.Grad)
			if err != nil {
				panic(err)
			}

			b.Grad, _ = b.Grad.Add(gradB)

		}
	}

	return out
}

func Neg(a *Tensor) *Tensor {
	out := _unaryOp(a, a.Data.Rows, a.Data.Cols, types.None, types.OpNeg)

	val := matrix.NewEmptyMatrix[float64](a.Data.Rows, a.Data.Cols)

	val = a.Data.Map(func(v float64) float64 {
		return -v
	})

	out.Data = &val

	out.Backward = func() {
		if !a.HasFlag(types.RequiresGradFlag) {
			return
		}

		for row := 0; row < a.Data.Rows; row++ {
			for col := 0; col < a.Data.Cols; col++ {
				a.Grad.Set(row, col, a.Grad.
					At(row, col)+(-out.Grad.At(row, col)))
			}
		}
	}

	return out
}
func Transpose(a *Tensor) *Tensor {

	out := _unaryOp(a, a.Data.Rows, a.Data.Cols, types.None, types.OpTranspose)

	aT := a.Data.Transpose()
	out.Data = &aT

	out.Backward = func() {
		if !a.HasFlag(types.RequiresGradFlag) {
			return
		}

		gradT := out.Grad.Transpose()
		a.Grad, _ = a.Grad.Add(&gradT)
	}

	return out
}
func Exp(a *Tensor) *Tensor {

	out := _unaryOp(a, a.Data.Rows, a.Data.Cols, types.None, types.OpExp)

	val := matrix.NewEmptyMatrix[float64](a.Data.Rows, a.Data.Cols)

	val = a.Data.Map(func(v float64) float64 {
		return math.Exp(v)
	})

	out.Data = &val

	out.Backward = func() {

		if !a.HasFlag(types.RequiresGradFlag) {
			return
		}

		for row := 0; row < a.Data.Rows; row++ {
			for col := 0; col < a.Data.Cols; col++ {
				grad := out.Grad.At(row, col) * out.Data.At(row, col)
				a.Grad.Set(row, col, a.Grad.At(row, col)+grad)
			}
		}

	}

	return out
}
func Log(a *Tensor) *Tensor {

	out := _unaryOp(a, a.Data.Rows, a.Data.Cols, types.None, types.OpLog)

	val := matrix.NewEmptyMatrix[float64](a.Data.Rows, a.Data.Cols)

	val = a.Data.Map(func(v float64) float64 {
		return math.Log(v)
	})

	out.Data = &val

	out.Backward = func() {
		if !a.HasFlag(types.RequiresGradFlag) {
			return
		}

		for row := 0; row < a.Data.Rows; row++ {
			for col := 0; col < a.Data.Cols; col++ {
				grad := out.Grad.At(row, col) / a.Data.At(row, col)
				a.Grad.Set(row, col, a.Grad.At(row, col)+grad)
			}
		}
	}

	return out
}
func Pow(a *Tensor, power float64) *Tensor {

	out := _unaryOp(a, a.Data.Rows, a.Data.Cols, types.None, types.OpPow)

	val := matrix.NewEmptyMatrix[float64](a.Data.Rows, a.Data.Cols)

	val = a.Data.Map(func(v float64) float64 {
		return math.Pow(v, power)
	})

	out.Data = &val

	out.Backward = func() {
		if !a.HasFlag(types.RequiresGradFlag) {
			return
		}

		for row := 0; row < a.Data.Rows; row++ {
			for col := 0; col < a.Data.Cols; col++ {

				x := a.Grad.At(row, col)
				localGrad := power * math.Pow(x, power-1)
				grad := out.Grad.At(row, col) * localGrad

				a.Grad.Set(row, col, a.Grad.At(row, col)+grad)
			}
		}
	}

	return out
}
func Sum(a *Tensor) *Tensor {

	out := _unaryOp(a, 1, 1, types.None, types.OpSum)

	val := NewScalar[float64](0.0)

	sum := 0.0

	for row := 0; row < a.Data.Rows; row++ {
		for col := 0; col < a.Data.Cols; col++ {
			sum += a.Data.At(row, col)
		}
	}

	val.Set(0, 0, sum)
	out.Data = val

	out.Backward = func() {
		if !a.HasFlag(types.RequiresGradFlag) {
			return
		}

		grad := out.Grad.At(0, 0)

		for row := 0; row < a.Data.Rows; row++ {
			for col := 0; col < a.Data.Cols; col++ {
				a.Grad.Set(row, col, a.Grad.At(row, col)+grad)
			}
		}
	}

	return out
}

func Mean(a *Tensor) *Tensor {

	out := _unaryOp(a, 1, 1, types.None, types.OpMean)

	val := NewScalar[float64](0.0)

	sum := 0.0

	count := float64(a.Data.Rows * a.Data.Cols)

	for row := 0; row < a.Data.Rows; row++ {
		for col := 0; col < a.Data.Cols; col++ {
			sum += a.Data.At(row, col)
		}
	}

	val.Set(0, 0, sum/count)
	out.Data = val

	out.Backward = func() {

		if !a.HasFlag(types.RequiresGradFlag) {
			return
		}

		grad := out.Grad.At(0, 0)

		for row := 0; row < a.Data.Rows; row++ {
			for col := 0; col < a.Data.Cols; col++ {
				a.Grad.Set(row, col, a.Grad.At(row, col)+grad)
			}
		}

	}

	return out
}

func Mul(a, b *Tensor) *Tensor {

	matrix.AssertIsSameShape[float64](*a.Data, *b.Data)

	rows, cols := broadcastShape(a.Data, b.Data)

	out := _binaryOp(a, b, a.Data.Rows, a.Data.Cols, types.None, types.OpMul)

	val := matrix.NewEmptyMatrix[float64](rows, cols)

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			aVal := ValueAtBroadcast(a.Data, row, col)
			bVal := ValueAtBroadcast(b.Data, row, col)

			val.Set(row, col, aVal*bVal)
		}
	}

	out.Data = &val

	out.Backward = func() {
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				upstream := out.Grad.At(row, col)

				aVal := ValueAtBroadcast(a.Data, row, col)
				bVal := ValueAtBroadcast(b.Data, row, col)

				if a.HasFlag(types.RequiresGradFlag) {
					gradA := upstream * bVal
					AddGradBroadcast(a, row, col, gradA)
				}

				if b.HasFlag(types.RequiresGradFlag) {
					gradB := upstream * aVal
					AddGradBroadcast(b, row, col, gradB)
				}
			}
		}
	}

	return out
}
func Div(a, b *Tensor) *Tensor {

	if !IsScalar(b.Data) {
		matrix.AssertIsSameShape[float64](*a.Data, *b.Data)
	}

	rows, cols := a.Data.Rows, a.Data.Cols

	if IsScalar(b.Data) {
		rows, cols = a.Data.Rows, a.Data.Cols
	}

	if IsScalar(a.Data) {
		rows, cols = b.Data.Rows, b.Data.Cols
	}

	out := _binaryOp(a, b, a.Data.Rows, a.Data.Cols, types.None, types.OpDiv)

	val := matrix.NewEmptyMatrix[float64](rows, cols)

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			aVal := ValueAtBroadcast(a.Data, row, col)
			bVal := ValueAtBroadcast(b.Data, row, col)

			if bVal == 0 {
				panic("division by zero")
			}

			val.Set(row, col, aVal/bVal)
		}
	}

	out.Data = &val

	out.Backward = func() {
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				upstream := out.Grad.At(row, col)

				aVal := ValueAtBroadcast(a.Data, row, col)
				bVal := ValueAtBroadcast(b.Data, row, col)

				if a.HasFlag(types.RequiresGradFlag) {
					gradA := upstream / bVal
					AddGradBroadcast(a, row, col, gradA)
				}

				if b.HasFlag(types.RequiresGradFlag) {
					gradB := -upstream * aVal / (bVal * bVal)
					AddGradBroadcast(b, row, col, gradB)
				}
			}
		}
	}
	return out
}
func Maximum(a, b *Tensor) *Tensor {

	matrix.AssertIsSameShape[float64](*a.Data, *b.Data)

	out := _binaryOp(a, b, a.Data.Rows, a.Data.Cols, types.None, types.OpMaximum)

	val, _ := a.Data.Maximum(b.Data)

	out.Data = val

	out.Backward = func() {
		for row := 0; row < a.Data.Rows; row++ {
			for col := 0; col < a.Data.Cols; col++ {
				upstream := out.Grad.At(row, col)

				av := a.Data.At(row, col)
				bv := b.Data.At(row, col)

				aGradVal := a.Grad.At(row, col)
				bGradVal := b.Grad.At(row, col)

				if av > bv {
					if a.HasFlag(types.RequiresGradFlag) {
						a.Grad.Set(row, col, aGradVal+upstream)
					}
				} else if bv > av {
					if b.HasFlag(types.RequiresGradFlag) {
						b.Grad.Set(row, col, bGradVal+upstream)
					}
				} else {

					if a.HasFlag(types.RequiresGradFlag) {
						a.Grad.Set(row, col, aGradVal+upstream*0.5)
					}

					if b.HasFlag(types.RequiresGradFlag) {
						b.Grad.Set(row, col, bGradVal+upstream*0.5)
					}

				}

			}
		}
	}

	return out
}

func Max(a *Tensor) *Tensor {

	out := _unaryOp(a, 1, 1, types.None, types.OpMax)

	val := Scalar(0)

	maxVal := a.Data.At(0, 0)
	count := 0

	for row := 0; row < a.Data.Rows; row++ {
		for col := 0; col < a.Data.Cols; col++ {

			x := a.Data.At(row, col)

			if x > maxVal {
				maxVal = x
				count = 1
			} else if x == maxVal {
				count++
			}
		}
	}

	val.Data.Set(0, 0, maxVal)
	out.Data = val.Data

	out.Backward = func() {
		if !a.HasFlag(types.RequiresGradFlag) {
			return
		}

		upstream := out.Grad.At(0, 0)
		gradShare := upstream / float64(count)

		for row := 0; row < a.Data.Rows; row++ {
			for col := 0; col < a.Data.Cols; col++ {

				aGradVal := a.Grad.At(row, col)

				if a.Data.At(row, col) == maxVal {
					a.Grad.Set(row, col, aGradVal+gradShare)
				}
			}
		}
	}

	return out

}

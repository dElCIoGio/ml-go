package tensor

import (
	"ml/matrix"
)

func mapMatrix(
	m *matrix.Matrix[float64],
	fn func(float64) float64,
) *matrix.Matrix[float64] {
	out := matrix.NewEmptyMatrix[float64](m.Rows, m.Cols)

	for row := 0; row < m.Rows; row++ {
		for col := 0; col < m.Cols; col++ {
			out.Set(row, col, fn(m.At(row, col)))
		}
	}

	return &out
}

func elementwise(
	a *matrix.Matrix[float64],
	b *matrix.Matrix[float64],
	fn func(float64, float64) float64,
) *matrix.Matrix[float64] {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("matrices must have the same shape")
	}

	out := matrix.NewEmptyMatrix[float64](a.Rows, a.Cols)

	for row := 0; row < a.Rows; row++ {
		for col := 0; col < a.Cols; col++ {
			out.Set(row, col, fn(a.At(row, col), b.At(row, col)))
		}
	}

	return &out
}

func transposeMatrix(
	m *matrix.Matrix[float64],
) *matrix.Matrix[float64] {
	out := matrix.NewEmptyMatrix[float64](m.Cols, m.Rows)

	for row := 0; row < m.Rows; row++ {
		for col := 0; col < m.Cols; col++ {
			out.Set(col, row, m.At(row, col))
		}
	}

	return &out
}

func elementwiseBroadcast(
	a *matrix.Matrix[float64],
	b *matrix.Matrix[float64],
	fn func(float64, float64) float64,
) *matrix.Matrix[float64] {
	rows, cols := broadcastShape(a, b)

	out := matrix.NewEmptyMatrix[float64](rows, cols)

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			aVal := ValueAtBroadcast(a, row, col)
			bVal := ValueAtBroadcast(b, row, col)

			out.Set(row, col, fn(aVal, bVal))
		}
	}

	return &out
}

func ValueAtBroadcast(m *matrix.Matrix[float64], row, col int) float64 {
	if IsScalar(m) {
		return m.At(0, 0)
	}

	return m.At(row, col)
}

func IsScalar(m *matrix.Matrix[float64]) bool {
	return m.Rows == 1 && m.Cols == 1
}

func broadcastShape(a, b *matrix.Matrix[float64]) (int, int) {
	if a.Rows == b.Rows && a.Cols == b.Cols {
		return a.Rows, a.Cols
	}

	if IsScalar(a) {
		return b.Rows, b.Cols
	}

	if IsScalar(b) {
		return a.Rows, a.Cols
	}

	panic("matrices must have the same shape or one must be scalar")
}

func AddGradBroadcast(
	t *Tensor,
	row, col int,
	value float64,
) {
	if t.Grad == nil {
		return
	}

	if IsScalar(t.Grad) {
		t.Grad.Set(0, 0, t.Grad.At(0, 0)+value)
		return
	}

	t.Grad.Set(row, col, t.Grad.At(row, col)+value)
}

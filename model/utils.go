package model

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

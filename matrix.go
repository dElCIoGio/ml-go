package main

import (
	"errors"
	"fmt"
)

type Matrix[T Number] struct {
	Rows, Cols int
	Data       []Vector[T]
}

func NewEmptyMatrix[T Number](rows, cols int) Matrix[T] {

	var data []Vector[T]

	for i := 0; i < rows; i++ {
		data = append(data, NewEmptyVector[T](cols))
	}

	return Matrix[T]{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

func NewMatrix[T Number](data [][]T) Matrix[T] {
	if len(data) == 0 {
		panic("matrix cannot be empty")
	}

	cols := len(data[0])

	var rows []Vector[T]

	for _, row := range data {
		if len(row) != cols {
			panic("all rows must have the same length")
		}

		vector := NewVector[T](row)
		rows = append(rows, vector)
	}

	return Matrix[T]{
		Rows: len(data),
		Cols: cols,
		Data: rows,
	}
}

type transpose struct {
	T bool
}

type MatAtOption func(*transpose)

func WithTranspose() MatAtOption {
	return func(t *transpose) {
		t.T = true
	}
}
func (m Matrix[T]) At(row, col int, options ...MatAtOption) T {

	cfg := transpose{}

	for _, option := range options {
		option(&cfg)
	}

	if cfg.T {
		row, col = col, row
	}

	return m.Data[row].Data[col]
}
func (m Matrix[T]) Set(row, col int, v T) {
	m.Data[row].Data[col] = v
}

func (m Matrix[T]) Row(row int) Vector[T] {
	return m.Data[row]
}

func (m Matrix[T]) Col(col int) Vector[T] {
	data := make([]T, m.Rows)

	for i := 0; i < m.Rows; i++ {
		data[i] = m.Data[i].Data[col]
	}

	return Vector[T]{Data: data}
}

func (m Matrix[T]) Copy() Matrix[T] {
	return m
}

func (m Matrix[T]) Fill(val T) {
	for i1, row := range m.Data {
		for i2, _ := range row.Data {
			m.Set(i1, i2, val)
		}
	}
}

func (m Matrix[T]) Scale(scale T) {
	for _, row := range m.Data {
		row.MulScalar(scale)
	}
}

func (m Matrix[T]) Print() {
	for _, v := range m.Data {
		v.Print()
	}
}

func (m Matrix[T]) Add(other Matrix[T]) (Matrix[T], error) {
	var rows [][]T

	if m.Rows != other.Rows || m.Cols != other.Cols {
		return m, errors.New("The two matrices have to be of the same dimentions")
	}

	for i := 0; i < m.Rows; i++ {
		vector, _ := m.Data[i].Add(other.Data[i])
		rows = append(rows, vector.Data)
	}

	matrix := NewMatrix[T](rows)

	return matrix, nil

}

func (m Matrix[T]) Sub(other Matrix[T]) (Matrix[T], error) {
	var rows [][]T

	if m.Rows != other.Rows || m.Cols != other.Cols {
		return m, errors.New("The two matrices have to be of the same dimentions")
	}

	for i := 0; i < m.Rows; i++ {
		vector, _ := m.Data[i].Add(other.Data[i])
		rows = append(rows, vector.Data)
	}

	matrix := NewMatrix[T](rows)

	return matrix, nil

}

type MatMulOptions struct {
	TransposeA bool
	TransposeB bool
}

func (m Matrix[T]) Mul(other Matrix[T], options ...MatMulOptions) (Matrix[T], error) {
	opts := MatMulOptions{}

	if len(options) > 0 {
		opts = options[0]
	}

	aRows := m.Rows
	aCols := m.Cols

	bRows := other.Rows
	bCols := other.Cols

	if opts.TransposeA {
		aRows, aCols = aCols, aRows
	}

	if opts.TransposeB {
		bRows, bCols = bCols, bRows
	}

	if aCols != bRows {
		return Matrix[T]{}, fmt.Errorf(
			"cannot multiply matrices: A has %d columns, B has %d rows",
			aCols,
			bRows,
		)
	}

	result := NewEmptyMatrix[T](aRows, bCols)

	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			var sum T

			for k := 0; k < aCols; k++ {
				var aValue T
				var bValue T

				if opts.TransposeA {
					aValue = m.At(i, k, WithTranspose())
				} else {
					aValue = m.At(i, k)
				}

				if opts.TransposeB {
					bValue = other.At(k, j, WithTranspose())
				} else {
					bValue = other.At(k, j)
				}

				sum += aValue * bValue
			}

			result.Set(i, j, sum)
		}
	}

	return result, nil
}
func (m Matrix[T]) Transpose() Matrix[T] {

	matrix := NewEmptyMatrix[T](m.Cols, m.Rows)

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			matrix.Set(j, i, m.At(i, j))
		}
	}

	return matrix

}

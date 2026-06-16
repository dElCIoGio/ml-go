package matrix

import (
	"encoding/binary"
	"math/rand"
	"ml/types"
	"ml/vector"
	"os"
)

type Matrix[T types.Number] struct {
	Rows, Cols int
	Data       []vector.Vector[T]
}

func NewEmptyMatrix[T types.Number](rows, cols int) Matrix[T] {

	var data []vector.Vector[T]

	for i := 0; i < rows; i++ {
		data = append(data, vector.NewEmptyVector[T](cols))
	}

	return Matrix[T]{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

func NewMatrix[T types.Number](data [][]T) Matrix[T] {
	if len(data) == 0 {
		panic("matrix cannot be empty")
	}

	cols := len(data[0])

	var rows []vector.Vector[T]

	for _, row := range data {
		if len(row) != cols {
			panic("all rows must have the same length")
		}

		v := vector.NewVector[T](row)
		rows = append(rows, v)
	}

	return Matrix[T]{
		Rows: len(data),
		Cols: cols,
		Data: rows,
	}
}

func NewRandomMatrix[T types.Number](rows, cols int) Matrix[T] {

	m := NewEmptyMatrix[T](rows, cols)

	m.Map(func(v T) T {
		return T(rand.NormFloat64() * 0.01)
	})

	return m
}

func LoadMat[T types.Number](rows, cols int, filename string) (*Matrix[T], error) {
	mat := NewEmptyMatrix[T](rows, cols)

	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	flat := make([]T, rows*cols)

	if err := binary.Read(f, binary.LittleEndian, flat); err != nil {
		return nil, err
	}

	for r := int(0); r < rows; r++ {
		start := r * cols
		end := start + cols

		copy(mat.Data[r].Data, flat[start:end])
	}

	return &mat, nil
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

func (m Matrix[T]) Row(row int) vector.Vector[T] {
	return m.Data[row]
}

func (m Matrix[T]) Col(col int) vector.Vector[T] {
	data := make([]T, m.Rows)

	for i := 0; i < m.Rows; i++ {
		data[i] = m.Data[i].Data[col]
	}

	return vector.Vector[T]{Data: data}
}

func (m Matrix[T]) Copy() Matrix[T] {
	newMatrix := NewEmptyMatrix[T](m.Rows, m.Cols)
	for i, row := range m.Data {
		newMatrix.Data[i] = row.Copy()
	}
	return newMatrix
}

func (m Matrix[T]) Fill(val T) {
	for i1, row := range m.Data {
		for i2, _ := range row.Data {
			m.Set(i1, i2, val)
		}
	}
}

func (m Matrix[T]) Print() {
	for _, v := range m.Data {
		v.Print()
	}
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

func (m Matrix[T]) Map(fn func(T) T) Matrix[T] {
	result := NewEmptyMatrix[T](m.Rows, m.Cols)

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Set(i, j, fn(m.At(i, j)))
		}
	}

	return result
}

func (m Matrix[T]) Clear() {
	m.Fill(0)
}

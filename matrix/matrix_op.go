package matrix

import (
	"errors"
	"fmt"
)

func (m Matrix[T]) Add(other *Matrix[T]) (*Matrix[T], error) {
	var rows [][]T

	if m.Rows != other.Rows || m.Cols != other.Cols {
		return &m, errors.New("The two matrices have to be of the same dimentions")
	}

	for i := 0; i < m.Rows; i++ {
		vector, _ := m.Data[i].Add(other.Data[i])
		rows = append(rows, vector.Data)
	}

	matrix := NewMatrix[T](rows)

	return &matrix, nil

}

func (m Matrix[T]) Sub(other *Matrix[T]) (*Matrix[T], error) {
	var rows [][]T

	if m.Rows != other.Rows || m.Cols != other.Cols {
		return &m, errors.New("The two matrices have to be of the same dimentions")
	}

	for i := 0; i < m.Rows; i++ {
		vector, _ := m.Data[i].Add(other.Data[i])
		rows = append(rows, vector.Data)
	}

	matrix := NewMatrix[T](rows)

	return &matrix, nil

}

type MatMulOptions struct {
	TransposeA bool
	TransposeB bool
}

func (m Matrix[T]) MatMul(other *Matrix[T], options ...MatMulOptions) (*Matrix[T], error) {
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
		return &Matrix[T]{}, fmt.Errorf(
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

	return &result, nil
}

func (m Matrix[T]) Scale(scale T) {
	for _, row := range m.Data {
		row.MulScalar(scale)
	}
}

func (m Matrix[T]) Mul(other *Matrix[T]) (*Matrix[T], error) {

	if !IsSameShape(m, *other) {
		return nil, errors.New("The two matrices have to be of the same dimentions")
	}

	matrix := NewEmptyMatrix[T](m.Rows, m.Cols)

	for row := 0; row < matrix.Rows; row++ {
		for col := 0; col < matrix.Cols; col++ {
			matrix.Set(row, col, m.At(row, col)*other.At(row, col))
		}
	}

	return &matrix, nil
}

func (m Matrix[T]) Div(other *Matrix[T]) (*Matrix[T], error) {

	matrix := NewEmptyMatrix[T](m.Rows, m.Cols)

	for row := 0; row < matrix.Rows; row++ {
		for col := 0; col < matrix.Cols; col++ {
			matrix.Set(row, col, m.At(row, col)/other.At(row, col))
		}
	}

	return &matrix, nil

}

func (m Matrix[T]) Maximum(other *Matrix[T]) (*Matrix[T], error) {

	if !IsSameShape(m, *other) {
		return nil, errors.New("The two matrices have to be of the same dimentions")
	}

	matrix := NewEmptyMatrix[T](m.Rows, m.Cols)

	for row := 0; row < m.Rows; row++ {
		for col := 0; col < m.Cols; col++ {
			av := m.At(row, col)
			bv := other.At(row, col)

			if av > bv {
				matrix.Set(row, col, av)
			} else {
				matrix.Set(row, col, bv)
			}
		}
	}

	return &matrix, nil

}

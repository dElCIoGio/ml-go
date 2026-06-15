package functions

import (
	"errors"
	"math"
	"ml/matrix"
	"ml/vector"
)

type SigmoidActivation struct{}

func (s SigmoidActivation) Forward(m matrix.Matrix[float64]) matrix.Matrix[float64] {
	return m.Map(func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	})
}

func (s SigmoidActivation) GradMatrix(m matrix.Matrix[float64]) matrix.Matrix[float64] {
	return m.Map(func(x float64) float64 {
		v := 1 / (1 + math.Exp(-x))
		return v * (1 - v)
	})
}

type ReLUActivation struct{}

func (r ReLUActivation) Forward(m matrix.Matrix[float64]) matrix.Matrix[float64] {
	return m.Map(func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	})
}

func (r ReLUActivation) GradMatrix(m matrix.Matrix[float64]) matrix.Matrix[float64] {
	return m.Map(func(x float64) float64 {
		if x > 0 {
			return 1
		}
		return 0
	})
}

func (r ReLUActivation) AddGrad(
	out *matrix.Matrix[float64],
	in *matrix.Matrix[float64],
	grad *matrix.Matrix[float64]) error {
	if out.Rows != in.Rows || out.Cols != in.Cols {
		return errors.New("the out and in matrices have to be of the same size")
	}

	if out.Rows != grad.Rows || out.Cols != grad.Cols {
		return errors.New("the out and grad matrices have to be of the same size")
	}

	for row := 0; row < out.Rows; row++ {
		for col := 0; col < out.Cols; col++ {
			if in.At(row, col) >= 0 {
				newGrad := out.At(row, col) + grad.At(row, col)
				out.Set(row, col, newGrad)
			}
		}
	}

	return nil
}

type SoftmaxActivation struct {
}

func (a *SoftmaxActivation) Pred(m matrix.Matrix[float64]) matrix.Matrix[float64] {

	newMatrix := matrix.NewEmptyMatrix[float64](m.Rows, m.Cols)

	for i, row := range m.Data {

		var sum float64
		var exp vector.Vector[float64]

		for _, val := range row.Data {
			n := math.Exp(val)
			exp.Data = append(exp.Data, n)
			sum += n
		}

		norm := exp.Map(func(value float64) float64 {
			return value / sum
		})

		newMatrix.Data[i] = norm

	}

	return newMatrix

}

type CrossEntropyActivation struct {
	Name string
}

func (c CrossEntropyActivation) Loss(m, labels matrix.Matrix[float64]) float64 {

	var val float64

	for i, row := range m.Data {
		log := row.Map(func(value float64) float64 {
			return math.Log(value)
		})

		var mul vector.Vector[float64]
		for j, _ := range log.Data {
			mul.Data = append(mul.Data, log.Data[j]*labels.At(i, j))
		}

		sum := -mul.Sum()

		val += sum

	}

	return val / float64(m.Rows)
}

func ReLU() ReLUActivation {
	return ReLUActivation{}
}
func Sigmoid() SigmoidActivation {
	return SigmoidActivation{}
}
func Softmax() SoftmaxActivation {
	return SoftmaxActivation{}
}
func CrossEntropy() CrossEntropyActivation {

	return CrossEntropyActivation{
		Name: "CrossEntropy",
	}

}

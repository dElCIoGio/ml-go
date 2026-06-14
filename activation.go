package main

import (
	"math"
)

type SigmoidActivation struct{}

func (s SigmoidActivation) Forward(m Matrix[float64]) Matrix[float64] {
	return m.Map(func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	})
}

func (s SigmoidActivation) GradMatrix(m Matrix[float64]) Matrix[float64] {
	return m.Map(func(x float64) float64 {
		v := 1 / (1 + math.Exp(-x))
		return v * (1 - v)
	})
}

type ReLUActivation struct{}

func (r ReLUActivation) Forward(m Matrix[float64]) Matrix[float64] {
	return m.Map(func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	})
}

func (r ReLUActivation) GradMatrix(m Matrix[float64]) Matrix[float64] {
	return m.Map(func(x float64) float64 {
		if x > 0 {
			return 1
		}
		return 0
	})
}

type SoftmaxActivation struct {
}

func (a *SoftmaxActivation) Pred(m Matrix[float64]) Matrix[float64] {

	matrix := NewEmptyMatrix[float64](m.Rows, m.Cols)

	for i, row := range m.Data {

		var sum float64
		var exp Vector[float64]

		for _, val := range row.Data {
			n := math.Exp(val)
			exp.Data = append(exp.Data, n)
			sum += n
		}

		norm := exp.Map(func(value float64) float64 {
			return value / sum
		})

		matrix.Data[i] = norm

	}

	return matrix

}

type CrossEntropyActivation struct {
	Name string
}

func (c CrossEntropyActivation) Loss(m, labels Matrix[float64]) float64 {

	var val float64

	for i, row := range m.Data {
		log := row.Map(func(value float64) float64 {
			return math.Log(value)
		})

		var mul Vector[float64]
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

package main

import (
	"math"
)

type Activation struct {
	Name string

	Fn   func(float64) float64
	Grad func(float64) float64
}

func (a *Activation) Forward(m Matrix[float64]) Matrix[float64] {
	return m.Map(a.Fn)
}

func (a *Activation) GradMatrix(m Matrix[float64]) Matrix[float64] {
	return m.Map(a.Grad)
}

func ReLU() Activation {
	return Activation{
		Name: "relu",

		Fn: func(x float64) float64 {
			if x > 0 {
				return x
			}
			return 0
		},

		Grad: func(x float64) float64 {
			if x > 0 {
				return 1
			}
			return 0
		},
	}
}
func Sigmoid() Activation {
	return Activation{
		Name: "sigmoid",

		Fn: func(x float64) float64 {
			return 1 / (1 + math.Exp(-x))
		},

		Grad: func(x float64) float64 {
			s := 1 / (1 + math.Exp(-x))
			return s * (1 - s)
		},
	}
}

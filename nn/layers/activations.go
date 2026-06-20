package layers

import (
	"ml/nn/functions"
	"ml/tensor"
)

type ReLU struct{}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(input *tensor.Tensor) *tensor.Tensor {
	return functions.ReLU(input)
}

func (r *ReLU) Parameters() []*tensor.Tensor {
	return nil
}

type Sigmoid struct{}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(input *tensor.Tensor) *tensor.Tensor {
	return functions.Sigmoid(input)
}

func (s *Sigmoid) Parameters() []*tensor.Tensor {
	return nil
}

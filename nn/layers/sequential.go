package layers

import (
	"ml/tensor"
)

type Sequential struct {
	Layers []Layer
}

func NewSequential(layers ...Layer) *Sequential {
	return &Sequential{
		Layers: layers,
	}
}

func (s *Sequential) Forward(input *tensor.Tensor) *tensor.Tensor {
	out := input

	for _, layer := range s.Layers {
		out = layer.Forward(out)
	}

	return out
}

func (s *Sequential) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor

	for _, layer := range s.Layers {
		params = append(params, layer.Parameters()...)
	}

	return params
}

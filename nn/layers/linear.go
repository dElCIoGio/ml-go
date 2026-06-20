package layers

import (
	"ml/matrix"
	"ml/tensor"
	"ml/types"
)

type Linear struct {
	InFeatures  int
	OutFeatures int

	Weights *tensor.Tensor
	Bias    *tensor.Tensor
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	wData := matrix.RandomXavier(inFeatures, outFeatures)
	bData := matrix.NewEmptyMatrix[float64](1, outFeatures)

	weights := tensor.NewTensor(wData)
	weights.WithGrad()
	weights.AddFlag(types.ParameterFlag)

	bias := tensor.NewTensor(&bData)
	bias.WithGrad()
	bias.AddFlag(types.ParameterFlag)

	return &Linear{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		Weights:     weights,
		Bias:        bias,
	}
}

func (l *Linear) Forward(input *tensor.Tensor) *tensor.Tensor {
	return tensor.Add(
		tensor.MatMul(input, l.Weights),
		l.Bias,
	)
}

func (l *Linear) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{
		l.Weights,
		l.Bias,
	}
}

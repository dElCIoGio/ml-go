package layers

import (
	"ml/tensor"
)

type Layer interface {
	Forward(input *tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
}

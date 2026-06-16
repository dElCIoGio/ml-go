package functions

import (
	"ml/matrix"
	"ml/tensor"
	"ml/types"
)

func ReLU(x *tensor.Tensor) *tensor.Tensor {

	data := matrix.NewEmptyMatrix[float64](x.Data.Rows, x.Data.Cols)
	t := tensor.NewTensor(&data, tensor.Flags(types.RequiresGradFlag))

	return tensor.Maximum(t, x)

}
func Sigmoid(a *tensor.Tensor) *tensor.Tensor {
	m := matrix.NewEmptyMatrix[float64](a.Data.Rows, a.Data.Cols)
	m.Fill(1.0)
	ones := tensor.NewTensor(&m)

	neg := tensor.Neg(a)
	exp := tensor.Exp(neg)
	demon := tensor.Add(ones, exp)
	div := tensor.Div(ones, demon)
	return div
}
func Softmax(a *tensor.Tensor) *tensor.Tensor {

	exps := tensor.Exp(a)
	sum := tensor.Sum(exps)

	return tensor.Div(exps, sum)
}
func Tanh(x *tensor.Tensor) *tensor.Tensor {

	oneData := matrix.NewEmptyMatrix[float64](x.Data.Rows, x.Data.Cols)
	oneData.Fill(float64(1))

	twoData := matrix.NewEmptyMatrix[float64](x.Data.Rows, x.Data.Cols)
	twoData.Fill(float64(2))

	one := tensor.NewTensor(&oneData)
	two := tensor.NewTensor(&twoData)

	return tensor.Sub(
		tensor.Mul(two, Sigmoid(tensor.Mul(two, x))), one)
}

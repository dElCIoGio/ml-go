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
func CrossEntropy(pred, target *tensor.Tensor) *tensor.Tensor {

	m := matrix.NewEmptyMatrix[float64](pred.Data.Rows, pred.Data.Cols)
	m.Fill(1e-12)
	eps := tensor.NewTensor(&m)

	safePred := tensor.Add(pred, eps)
	logPred := tensor.Log(safePred)
	mul := tensor.Mul(target, logPred)
	sum := tensor.Sum(tensor.Neg(mul))

	return sum

}

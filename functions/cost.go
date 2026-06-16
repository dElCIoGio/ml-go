package functions

import (
	"ml/matrix"
	"ml/tensor"
)

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

func MSE(pred, target *tensor.Tensor) *tensor.Tensor {
	diff := tensor.Sub(pred, target)
	squared := tensor.Pow(diff, 2)
	return tensor.Mean(squared)
}

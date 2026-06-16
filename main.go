package main

import (
	"fmt"
	"ml/functions"
	"ml/matrix"
	"ml/tensor"
	"ml/vector"
)

func DrawMNISTDigit(data vector.Vector[float32]) {
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			num := data.Data[x+y*28]

			if num < 0 {
				num = 0
			}
			if num > 1 {
				num = 1
			}

			col := 232 + int(num*23)

			fmt.Printf("\x1b[48;5;%dm  ", col)
		}

		fmt.Println()
	}

	fmt.Print("\x1b[0m")
}

func namedTensor(name string, names map[*tensor.Tensor]string, inputs ...*tensor.Tensor) *tensor.Tensor {
	t := &tensor.Tensor{
		Inputs: inputs,
	}

	names[t] = name
	return t
}

func main() {

	//trainImages, _ := LoadMat[float32](60000, 784, "data/train_images.mat")
	//testImages, _ := LoadMat[float32](10000, 784, "data/test_images.mat")

	//trainLabels := NewEmptyMatrix[float32](60000, 10)
	//testLabels := NewEmptyMatrix[float32](10000, 10)
	//
	//trainLabelsFiles, _ := LoadMat[float32](60000, 1, "data/train_labels.mat")
	//testLabelsFiles, _ := LoadMat[float32](10000, 1, "data/test_labels.mat")
	//
	//for i := 0; i < 60000; i++ {
	//	num := trainLabelsFiles.At(i, 0)
	//	trainLabels.Set(i, int(num), 1)
	//}
	//
	//for i := 0; i < 10000; i++ {
	//	num := testLabelsFiles.At(i, 0)
	//	testLabels.Set(i, int(num), 1)
	//}
	data := matrix.NewMatrix[float64]([][]float64{
		{1, 2, 3},
	})

	x := tensor.NewTensor(&data)
	x.WithGrad()

	targetData := matrix.NewMatrix[float64]([][]float64{
		{0, 0, 1},
	})

	target := tensor.NewTensor(&targetData)

	pred := functions.Softmax(x)
	loss := functions.CrossEntropy(pred, target)

	prog := tensor.ModelProgramCreate(loss)

	prog.Compute()
	prog.ComputeGrads()

	fmt.Println("softmax output:")
	fmt.Println(pred.Data)

	fmt.Println("loss:")
	fmt.Println(loss.Data)

	fmt.Println("x grad:")
	fmt.Println(x.Grad)

}

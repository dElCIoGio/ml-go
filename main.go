package main

import (
	"fmt"
	"ml/functions"
	"ml/matrix"
	"ml/tensor"
	"ml/vector"
)

func DrawMNISTDigit(data vector.Vector[float64]) {
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

func main() {

	trainImagesMatrix, _ := matrix.LoadMat[float64](60000, 784, "data/train_images.mat")
	//testImagesMatrix, _ := matrix.LoadMat[float64](10000, 784, "data/test_images.mat")

	trainLabelsMatrix := matrix.NewEmptyMatrix[float64](60000, 10)
	testLabelsMatrix := matrix.NewEmptyMatrix[float64](10000, 10)

	trainLabelsFilesMatrix, _ := matrix.LoadMat[float64](60000, 1, "data/train_labels.mat")
	testLabelsFilesMatrix, _ := matrix.LoadMat[float64](10000, 1, "data/test_labels.mat")

	for i := 0; i < 60000; i++ {
		num := trainLabelsFilesMatrix.At(i, 0)
		trainLabelsMatrix.Set(i, int(num), 1)
	}

	for i := 0; i < 10000; i++ {
		num := testLabelsFilesMatrix.At(i, 0)
		testLabelsMatrix.Set(i, int(num), 1)
	}

	//
	//testImages := tensor.NewTensor(testImagesMatrix)
	//
	//testLabels := tensor.NewTensor(&testLabelsMatrix)

	x := tensor.NewTensor(trainImagesMatrix)
	y := tensor.NewTensor(&trainLabelsMatrix)

	wData := matrix.NewRandomMatrix[float64](784, 10)
	w := tensor.NewTensor(&wData)
	w.WithGrad()

	logits := tensor.MatMul(x, w)
	pred := functions.Softmax(logits)
	loss := functions.CrossEntropy(pred, y)

	prog := tensor.ModelProgramCreate(loss)

	prog.Compute()
	prog.ComputeGrads()

	fmt.Println("prediction:")
	fmt.Println(pred.Data)

	fmt.Println("loss:")
	fmt.Println(loss.Data)

	fmt.Println("W grad:")
	fmt.Println(w.Grad)

}

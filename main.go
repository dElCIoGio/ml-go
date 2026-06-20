package main

import (
	"fmt"
	"ml/matrix"
	functions2 "ml/nn/functions"
	"ml/nn/layers"
	"ml/tensor"
	"ml/types"
	"ml/vector"
)

func UpdateParameters(prog *tensor.ModelProgram, lr float64) {
	for _, t := range prog.Vars {
		if !t.HasFlag(types.ParameterFlag) {
			continue
		}

		if t.Grad == nil {
			continue
		}

		for row := 0; row < t.Data.Rows; row++ {
			for col := 0; col < t.Data.Cols; col++ {
				newValue := t.Data.At(row, col) - lr*t.Grad.At(row, col)
				t.Data.Set(row, col, newValue)
			}
		}
	}
}

func ArgMax(m *matrix.Matrix[float64]) int {
	maxIndex := 0
	maxValue := m.At(0, 0)

	for col := 1; col < m.Cols; col++ {
		v := m.At(0, col)

		if v > maxValue {
			maxValue = v
			maxIndex = col
		}
	}

	return maxIndex
}

func SetRow(dst *matrix.Matrix[float64], src *matrix.Matrix[float64], rowIndex int) {
	for col := 0; col < src.Cols; col++ {
		dst.Set(0, col, src.At(rowIndex, col))
	}
}

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

func Evaluate(
	prog *tensor.ModelProgram,
	input *tensor.Tensor,
	target *tensor.Tensor,
	pred *tensor.Tensor,
	loss *tensor.Tensor,
	images *matrix.Matrix[float64],
	labels *matrix.Matrix[float64],
	limit int,
) {
	correct := 0
	totalLoss := 0.0

	for i := 0; i < limit; i++ {
		SetRow(input.Data, images, i)
		SetRow(target.Data, labels, i)

		prog.Compute()

		totalLoss += loss.Data.At(0, 0)

		if ArgMax(pred.Data) == ArgMax(target.Data) {
			correct++
		}
	}

	fmt.Printf(
		"Eval accuracy: %d/%d (%.2f%%), avg loss: %.4f\n",
		correct,
		limit,
		float64(correct)/float64(limit)*100,
		totalLoss/float64(limit),
	)
}

func updateParam(p *tensor.Tensor, lr float64) {
	for row := 0; row < p.Data.Rows; row++ {
		for col := 0; col < p.Data.Cols; col++ {
			p.Data.Set(row, col, p.Data.At(row, col)-lr*p.Grad.At(row, col))
		}
	}
}

func main() {

	trainImagesMatrix, _ := matrix.LoadMat[float64](60000, 784, "data/train_images.mat")
	testImagesMatrix, _ := matrix.LoadMat[float64](10000, 784, "data/test_images.mat")

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

	inputData := matrix.NewEmptyMatrix[float64](1, 784)
	targetData := matrix.NewEmptyMatrix[float64](1, 10)

	input := tensor.NewTensor(&inputData)
	target := tensor.NewTensor(&targetData)

	wData := matrix.RandomXavier(784, 10)

	weights := tensor.NewTensor(wData)
	weights.WithGrad()
	weights.AddFlag(types.ParameterFlag)

	nn := layers.NewSequential(
		layers.NewLinear(784, 128),
		layers.NewReLU(),
		layers.NewLinear(128, 10),
	)

	logits := nn.Forward(input)
	pred := functions2.Softmax(logits)
	loss := functions2.CrossEntropy(pred, target)

	model := tensor.ModelCreate()
	model.Input = input
	model.Output = pred
	model.DesiredOutput = target
	model.Cost = loss

	model.Compile()

	// 1. Tiny training loop
	fmt.Println("\n---- SMALL TRAINING TEST ----")

	desc := &tensor.ModelTrainingDesc{
		TrainImages: trainImagesMatrix,
		TestImages:  testImagesMatrix,
		TrainLabels: &trainLabelsMatrix,
		TestLabels:  &testLabelsMatrix,

		Epochs:       10,
		BatchSize:    1,
		LearningRate: 0.01,

		TrainLimit: 1000,
		TestLimit:  1000,
	}

	// 8. Evaluate before training
	fmt.Println("---- BEFORE TRAINING ----")
	model.Evaluate(testImagesMatrix, &testLabelsMatrix, desc.TestLimit)

	// 9. Train model
	fmt.Println("\n---- TRAINING ----")
	model.Train(desc)

	// 10. Evaluate after training
	fmt.Println("\n---- AFTER TRAINING ----")
	model.Evaluate(testImagesMatrix, &testLabelsMatrix, desc.TestLimit)

	// 11. Show individual predictions
	fmt.Println("\n---- SAMPLE PREDICTIONS ----")

	prog := model.ForwardProgram

	for i := 0; i < 5; i++ {
		PredictOne(prog, input, target, pred, testImagesMatrix, &testLabelsMatrix, i)
	}
}

func PredictOne(
	prog *tensor.ModelProgram,
	input *tensor.Tensor,
	target *tensor.Tensor,
	pred *tensor.Tensor,
	images *matrix.Matrix[float64],
	labels *matrix.Matrix[float64],
	index int,
) {
	SetRow(input.Data, images, index)
	SetRow(target.Data, labels, index)

	prog.Compute()

	predicted := ArgMax(pred.Data)
	actual := ArgMax(target.Data)

	fmt.Println("Image index:", index)
	fmt.Println("Prediction probabilities:")
	fmt.Println(pred.Data)

	fmt.Println("Predicted class:", predicted)
	fmt.Println("Actual class:", actual)

	if predicted == actual {
		fmt.Println("Correct ✅")
	} else {
		fmt.Println("Wrong ❌")
	}
}

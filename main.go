package main

import (
	"fmt"
	"ml/model"
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

	names := map[*tensor.Tensor]string{}

	x := namedTensor("x", names)

	w1 := namedTensor("w1", names)
	b1 := namedTensor("b1", names)

	w2 := namedTensor("w2", names)
	b2 := namedTensor("b2", names)

	target := namedTensor("target", names)

	mm1 := namedTensor("mm1 = MatMul(x, w1)", names, x, w1)
	h1 := namedTensor("h1 = Add(mm1, b1)", names, mm1, b1)

	// Shared input: h1 is used twice
	sq := namedTensor("sq = Mul(h1, h1)", names, h1, h1)

	mm2 := namedTensor("mm2 = MatMul(sq, w2)", names, sq, w2)

	// Skip connection: x is reused here
	skip := namedTensor("skip = Add(x, b2)", names, x, b2)

	out := namedTensor("out = Add(mm2, skip)", names, mm2, skip)

	diff := namedTensor("diff = Sub(out, target)", names, out, target)
	loss := namedTensor("loss = Mean(diff)", names, diff)

	prog := model.ModelProgramCreate(loss)

	for i, t := range prog.Vars {
		fmt.Printf("%d: %s\n", i, names[t])
	}

}

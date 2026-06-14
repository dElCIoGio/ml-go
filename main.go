package main

import (
	"fmt"
)

func DrawMNISTDigit(data Vector[float32]) {
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

	//trainImages, _ := LoadMat[float32](60000, 784, "data/train_images.mat")
	testImages, err := LoadMat[float32](10000, 784, "data/test_images.mat")
	if err != nil {
		fmt.Println(err)
	}

	DrawMNISTDigit(testImages.Data[4])

}

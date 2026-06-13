package main

func main() {

	relu := ReLU()

	z := NewMatrix[float64]([][]float64{
		{-1, 2, 3},
		{4, -5, 6},
	})

	grad := relu.GradMatrix(z)
	grad.Print()
}

package vector

import (
	"fmt"
	"ml/types"
)

type Vector[T types.Number] struct {
	Data []T
}

func NewVector[T types.Number](data []T) Vector[T] {
	return Vector[T]{Data: data}
}

func NewEmptyVector[T types.Number](len int) Vector[T] {

	if len < 0 {
		panic("Vector length must be positive!")
	}

	return Vector[T]{Data: make([]T, len)}
}

func (v Vector[T]) Fill(val T) {
	for i := 0; i < v.Len(); i++ {
		v.Data[i] = val
	}
}

func (v Vector[T]) At(i int) T {
	return v.Data[i]
}

func (v Vector[T]) Len() int {
	return len(v.Data)
}

func (v Vector[T]) Print() {
	builder := ""
	for _, val := range v.Data {
		builder += fmt.Sprintf("%v\t", val)
	}
	fmt.Println(builder)
}

func (v Vector[T]) Map(fn func(T) T) Vector[T] {

	vector := NewEmptyVector[T](v.Len())

	for i, _ := range v.Data {
		vector.Data[i] = fn(v.Data[i])
	}

	return vector
}

func (v Vector[T]) Copy() Vector[T] {
	vector := NewEmptyVector[T](v.Len())
	copy(vector.Data, v.Data)
	return vector
}

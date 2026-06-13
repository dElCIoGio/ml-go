package main

import (
	"errors"
	"fmt"
)

type Vector[T Number] struct {
	Data []T
}

func NewVector[T Number](data []T) Vector[T] {
	return Vector[T]{Data: data}
}

func NewEmptyVector[T Number](len int) Vector[T] {

	if len < 0 {
		panic("Vector length must be positive!")
	}

	return Vector[T]{Data: make([]T, len)}
}

func (v Vector[T]) Add(other Vector[T]) (Vector[T], error) {

	vector := Vector[T]{
		Data: make([]T, len(v.Data)),
	}

	if v.Len() != other.Len() {
		return Vector[T]{}, errors.New("Vector lengths are different")
	}

	for i := 0; i < v.Len(); i++ {
		val := v.Data[i] + other.Data[i]
		vector.Data[i] = val
	}

	return vector, nil
}

func (v Vector[T]) Sub(other Vector[T]) (Vector[T], error) {
	vector := Vector[T]{
		Data: make([]T, len(v.Data)),
	}

	if v.Len() != other.Len() {
		return Vector[T]{}, errors.New("Vector lengths are different")
	}

	for i := 0; i < v.Len(); i++ {
		val := v.Data[i] - other.Data[i]
		vector.Data[i] = val
	}

	return vector, nil

}

func (v Vector[T]) Fill(val T) {
	for i := 0; i < v.Len(); i++ {
		v.Data[i] = val
	}
}

func (v Vector[T]) Dot(other Vector[T]) (T, error) {

	if v.Len() != other.Len() {
		return 0, errors.New("Vector lengths are different")
	}

	var dotProduct T

	for i := 0; i < v.Len(); i++ {
		dotProduct += v.Data[i] * other.Data[i]
	}

	return dotProduct, nil

}

func (v Vector[T]) MulScalar(s T) Vector[T] {
	for i := 0; i < v.Len(); i++ {
		v.Data[i] *= s
	}
	return v
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

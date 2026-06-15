package vector

import (
	"errors"
)

func (v Vector[T]) Sum() T {
	var sum T
	for _, val := range v.Data {
		sum += val
	}
	return sum
}

func (v Vector[T]) MulScalar(s T) Vector[T] {
	for i := 0; i < v.Len(); i++ {
		v.Data[i] *= s
	}
	return v
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

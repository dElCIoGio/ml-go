package main

type Number interface {
	~int | ~int64 | ~int32 | ~float64 | ~float32
}

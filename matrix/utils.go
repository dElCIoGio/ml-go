package matrix

import (
	"math"
	"math/bits"
	"ml/types"
)

func IsSameShape[T types.Number](a, b Matrix[T]) bool {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return false
	}
	return true
}

func AssertIsSameShape[T types.Number](a, b Matrix[T]) {
	if !IsSameShape(a, b) {
		panic("The matrices must be of the same shape")
	}
}

type State struct {
	state uint64
	inc   uint64
}

var global = State{
	state: 0x853c49e6748fea9b,
	inc:   0xda3e39cb94b95bdb,
}

func SeedR(rng *State, initState uint64, initSeq uint64) {
	rng.state = 0
	rng.inc = (initSeq << 1) | 1

	RandR(rng)

	rng.state += initState

	RandR(rng)
}

func Seed(initState uint64, initSeq uint64) {
	SeedR(&global, initState, initSeq)
}

func RandR(rng *State) uint64 {
	oldState := rng.state

	rng.state = oldState*6364136223846793005 + rng.inc

	xorShifted := uint64(((oldState >> 18) ^ oldState) >> 27)
	rot := uint(oldState >> 59)

	return bits.RotateLeft64(xorShifted, -int(rot))
}

func Rand() uint64 {
	return RandR(&global)
}

func RandFloat32R(rng *State) float32 {
	return float32(RandR(rng)) / float32(math.MaxUint32)
}

func RandFloat64R(rng *State) float64 {
	return float64(RandR(rng)) / float64(math.MaxUint64)
}

func RandFloat32() float32 {
	return RandFloat32R(&global)
}

func RandFloat64() float64 {
	return RandFloat64R(&global)
}

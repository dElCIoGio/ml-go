# ml-go

A small machine learning library written in Go from scratch.

This project is mainly a learning-focused implementation of the core ideas behind neural networks: vectors, matrices, tensors, computation graphs, automatic differentiation, activation functions, losses, and simple training loops.

It is not intended to replace production machine learning frameworks. The goal is to understand what those frameworks are doing internally by building the pieces directly in Go.

## Current status

This project is still early-stage and the API may change as the library becomes more organised.

At the moment, the repository includes:

- Generic vector and matrix primitives
- Matrix creation, copying, transposition, mapping, filling, and random initialisation
- Xavier-style random matrix initialisation
- Binary matrix loading for local dataset files
- Tensor objects with data, gradients, operations, input tracking, and flags
- A simple computation graph/program abstraction for forward and backward passes
- Basic neural-network functions such as ReLU, Sigmoid, Softmax, Tanh, and Cross Entropy
- A small MNIST-style linear classifier experiment in `main.go`

## Project structure

```text
.
├── matrix/          # Matrix data structures and matrix utilities
├── vector/          # Vector data structures and vector utilities
├── tensor/          # Tensor objects, graph operations, and gradient computation
├── nn/functions/    # Neural-network activation and loss functions
├── types/           # Shared type constraints, flags, and operation types
├── main.go          # Current MNIST-style training experiment
└── go.mod
```

## Requirements

- Go 1.18 or newer

The module currently uses Go generics, so Go 1.18+ is required.

## Running the project

From the root of the repository:

```bash
go run .
```

The current demo in `main.go` expects local MNIST-style data files inside a `data/` directory:

```text
data/train_images.mat
data/train_labels.mat
data/test_images.mat
data/test_labels.mat
```

The `data/` folder is ignored by Git, so those files need to exist locally before running the MNIST experiment.

## Minimal example

The current API can be used to build a tiny computation graph, run a forward pass, and compute gradients:

```go
package main

import (
    "fmt"

    "ml/matrix"
    functions "ml/nn/functions"
    "ml/tensor"
    "ml/types"
)

func main() {
    inputData := matrix.NewMatrix[float64]([][]float64{
        {1, 2, 3},
    })

    targetData := matrix.NewMatrix[float64]([][]float64{
        {0, 0, 1},
    })

    input := tensor.NewTensor(&inputData)
    target := tensor.NewTensor(&targetData)

    weightsData := matrix.RandomXavier(3, 3)
    weights := tensor.NewTensor(weightsData)
    weights.WithGrad()
    weights.AddFlag(types.RequiresGradFlag)
    weights.AddFlag(types.ParameterFlag)

    logits := tensor.MatMul(input, weights)
    prediction := functions.Softmax(logits)
    loss := functions.CrossEntropy(prediction, target)

    program := tensor.ModelProgramCreate(loss)
    program.Compute()
    program.ComputeGrads()

    fmt.Println("prediction:")
    fmt.Println(prediction.Data)

    fmt.Println("loss:")
    fmt.Println(loss.Data)
}
```

## What this project is helping me learn

This library is being built to understand machine learning from the inside out, especially:

- How vectors and matrices support neural-network calculations
- How tensors store values and gradients
- How operations form a computation graph
- How a forward pass produces predictions and losses
- How a backward pass propagates gradients
- How parameter updates improve a model over time
- How a simple classifier can be trained without relying on a full ML framework

## Roadmap

Possible next steps:

- Move experiments from `main.go` into an `examples/` directory
- Add a cleaner public API for building models
- Add reusable layers such as Linear/Dense
- Add optimisers such as SGD, Momentum, and Adam
- Add tests for matrix, tensor, and gradient operations
- Add benchmarks for core matrix and tensor operations
- Improve broadcasting and numerical stability
- Add package-level documentation

## Disclaimer

This is an educational project. It is useful for learning how machine learning systems work internally, but it is not yet a production-ready ML library.

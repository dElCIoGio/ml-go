# ml-go

A machine learning library written in Go from scratch.

This is an educational project where I am rebuilding the core pieces behind neural networks instead of relying on existing ML frameworks. The goal is to understand how the lower-level parts work: vectors, matrices, tensors, computation graphs, gradients, activation functions, losses, and simple training loops.

## Current status

This repository is an early public snapshot of the project. The code is still being organised, and the full library structure is being developed incrementally.

The project is not production-ready yet. APIs, package names, and examples may change as the implementation becomes cleaner.

## What I am building

The long-term goal is to build a small but understandable ML library that includes:

- Vector and matrix primitives
- Tensor objects
- Basic tensor operations
- Computation graph execution
- Automatic differentiation
- Activation functions
- Loss functions
- Simple optimisers
- Small neural-network examples

The purpose is not to compete with libraries like TensorFlow or PyTorch. The purpose is to learn what those libraries do internally by implementing the ideas directly in Go.

## Why Go?

I am using Go because it is simple, explicit, compiled, and good for understanding systems-level behaviour. Building ML concepts in Go forces me to think carefully about memory, data structures, APIs, and how the training process actually works.

## Project direction

The project is currently moving towards a cleaner library-style structure. The intended direction is:

```text
.
├── matrix/          # Matrix data structures and operations
├── vector/          # Vector data structures and operations
├── tensor/          # Tensors, graph operations, and gradients
├── nn/              # Neural-network functions, layers, and losses
├── optim/           # Optimisers such as SGD or Adam
├── examples/        # Small examples and experiments
└── tests/           # Unit tests for core behaviour
```

Some of these packages may not exist yet in the public repository. They represent the direction the project is moving towards.

## Running the project

The repository currently uses Go modules.

```bash
go run .
```

Some experiments may require local data files that are not committed to the repository, especially dataset files such as MNIST exports. If an example expects a `data/` folder, those files need to be generated or added locally.

## Learning goals

This project is helping me understand:

- How numerical data is represented with vectors, matrices, and tensors
- How a forward pass moves data through a model
- How losses measure prediction error
- How gradients are calculated through backpropagation
- How parameters are updated during training
- How a simple neural network can be built without a full ML framework
- How to structure a Go project as a reusable library

## Roadmap

Planned improvements:

- Push and organise the latest local code
- Separate experiments from reusable library code
- Create cleaner public APIs for tensors and models
- Add reusable layers such as `Linear` or `Dense`
- Add optimisers such as SGD and Adam
- Add tests for matrix, tensor, and gradient operations
- Add examples for small datasets
- Improve package documentation

## Disclaimer

This is a work-in-progress learning project. The implementation is expected to change as I improve the design and continue learning how ML libraries are built internally.

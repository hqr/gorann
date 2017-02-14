# GoRANN [![][license-svg]][license-url]

Recurrent and Artificial Neural Networks

> The vanishing gradient problem of automatic differentiation or backpropagation in neural networks was partially overcome in 1992 by an early generative model called the neural history compressor, implemented as an unsupervised stack of recurrent neural networks... (Wikipedia)

## Overview

This is a from-scratch Golang implementation, inspired in part by the [Stanford ML course](http://cs229.stanford.edu/materials.html).

## Keywords

Gradient descent, SGD and mini-batching, L1/L2 regularization, Adagrad, Adadelta, RMSprop and ADAM optimization algorithms, hyper-parameters and other tunables, activation functions, input and output normalization.

Naive, fully connected ("unrolled") and partially connected ("limited") RNN, and more.

## Install

```
go get github.com/hqr/gorann
```

#### Test and run

See [Makefile](https://github.com/hqr/gorann/blob/master/Makefile) for test, lint, command-line help, and other useful targets.

[license-url]: https://github.com/hqr/gorann/blob/master/LICENSE
[license-svg]: https://img.shields.io/badge/license-MIT-blue.svg

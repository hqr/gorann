package gorann

import (
	"fmt"
	"math"
	"math/rand"
)

func ExampleF_xorbits() {
	rand.Seed(0) // for reproducible results

	// NN: input layer of 2 nodes, 2 hidden layers consisting of 16 nodes
	// and a single-node output layer that uses sigmoid() activation
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"sigmoid", 16}
	output := NeuLayerConfig{"sigmoid", 1}
	nn := NewNeuNetwork(input, hidden, 2, output, ADAM)
	nn.tunables.alpha = 0.4

	maxint := int32(0xff)
	normalize := func(vec []float64) {
		divElemVector(vec, float64(maxint))
	}
	denormalize := func(vec []float64) {
		mulElemVector(vec, float64(maxint))
	}
	nn.callbacks = NeuCallbacks{normalize, normalize, denormalize}

	xorbits := func(xvec []float64) []float64 {
		var y []float64 = []float64{0}
		a := int(xvec[0])
		b := int(xvec[1])
		y[0] = float64(a ^ b)
		return y
	}
	Xs := newMatrix(100, 2)
	converged := 0
	for converged == 0 {
		for i := 0; i < len(Xs); i++ {
			Xs[i][0] = float64(rand.Int31n(maxint))
			Xs[i][1] = float64(rand.Int31n(maxint))
		}
		converged = nn.Train(Xs, TrainParams{resultvalcb: xorbits, repeat: 3, testingpct: 60, maxcost: 1E-3})
	}
	// test and print the results (expected output below)
	var err, loss float64
	for k := 0; k < 4; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		y1 := nn.Predict(xvec)
		y2 := xorbits(xvec)
		err += nn.AbsError(y2)
		loss += nn.CostLinear(y2)
		a, b, c, d := int(xvec[0]), int(xvec[1]), int(y1[0]), int(y2[0])
		fmt.Printf("%08b ^ %08b -> %08b : %08b\n", a, b, c, d)
	}
	fmt.Printf("error %d, loss %.5f\n", int(err)/4, loss/4)
	// Output:
	// 10010001 ^ 01101100 -> 11111001 : 11111101
	// 01101101 ^ 10011110 -> 11101111 : 11110011
	// 11001011 ^ 11011110 -> 00011000 : 00010101
	// 10111101 ^ 00000011 -> 10111101 : 10111110
	// error 2, loss 0.00006
}

func ExampleF_sumsquares() {
	rand.Seed(0)
	// NN: input layer of 2 nodes, 2 hidden layers consisting of 16 nodes that utilize tanh(),
	// and a single-node output layer that uses sigmoid() activation
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"tanh", 20}
	output := NeuLayerConfig{"sigmoid", 1}
	nn := NewNeuNetwork(input, hidden, 2, output, Adadelta)

	sumsquares := func(xvec []float64) []float64 {
		var y []float64 = []float64{0}
		for i := 0; i < len(xvec); i++ {
			y[0] += xvec[i] * xvec[i]
		}
		return y
	}
	Xs := newMatrix(100, 2)
	converged := 0
	for converged == 0 {
		for i := 0; i < len(Xs); i++ {
			Xs[i][0] = rand.Float64() / 1.5
			Xs[i][1] = rand.Float64() / 1.5
		}
		converged = nn.Train(Xs, TrainParams{resultvalcb: sumsquares, repeat: 3, testingpct: 30, maxcost: 1E-6})
	}
	// use to estimate
	var loss float64
	for i := 0; i < 4; i++ {
		var xvec []float64 = []float64{0, 0}
		xvec[0] = rand.Float64() / 1.5
		xvec[1] = rand.Float64() / 1.5
		y1 := nn.Predict(xvec)
		y2 := sumsquares(xvec)
		loss += nn.CostLinear(y2)
		fmt.Printf("%.3f**2 + %.3f**2 -> %.3f : %.3f\n", xvec[0], xvec[1], y1[0], y2[0])
	}
	fmt.Printf("loss %.7f\n", loss/4.0)
	// Output:
	// 0.210**2 + 0.050**2 -> 0.042 : 0.046
	// 0.619**2 + 0.022**2 -> 0.382 : 0.384
	// 0.409**2 + 0.344**2 -> 0.283 : 0.285
	// 0.557**2 + 0.448**2 -> 0.512 : 0.511
	// loss 0.0000033
}

func ExampleF_sumlogarithms() {
	rand.Seed(1)
	// NN: input layer of 2 nodes, 4 hidden layers consisting of 8 tanh() nodes
	// and a single-node output using sigmoid() activation
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"tanh", 8}
	output := NeuLayerConfig{"tanh", 1}
	nn := NewNeuNetwork(input, hidden, 5, output, RMSprop)
	nn.tunables.momentum = 0.5
	nn.tunables.batchsize = 10

	normalize := func(vec []float64) {
		divElemVector(vec, float64(-8))
	}
	denormalize := func(vec []float64) {
		mulElemVector(vec, float64(-8))
	}
	nn.callbacks = NeuCallbacks{nil, normalize, denormalize}

	sumlogarithms := func(xvec []float64) []float64 {
		var y []float64 = []float64{0}
		for i := 0; i < len(xvec); i++ {
			y[0] += math.Log(xvec[i])
		}
		return y
	}
	Xs := newMatrix(100, 2)
	var converged int
	for converged&ConvergedWeight == 0 || converged&ConvergedGradient == 0 {
		for i := 0; i < len(Xs); i++ {
			Xs[i][0], Xs[i][1] = rand.Float64(), rand.Float64()
		}
		converged = nn.Train(Xs, TrainParams{resultvalcb: sumlogarithms, repeat: 3, maxweightdelta: 0.005, maxgradnorm: 0.01, maxbackprops: 4E6})
		if converged&ConvergedMaxBackprops > 0 {
			fmt.Printf("maxed out back propagations %d\n", nn.nbackprops)
			break
		}
	}

	// use to estimate
	var loss float64
	for i := 0; i < 4; i++ {
		var xvec []float64 = []float64{0, 0}
		xvec[0] = rand.Float64()
		xvec[1] = rand.Float64()
		y2 := sumlogarithms(xvec)
		y1 := nn.Predict(xvec)
		loss += nn.CostLinear(y2)
		fmt.Printf("log(%.3f) + log(%.3f) -> %.3f : %.3f\n", xvec[0], xvec[1], y1[0], y2[0])
	}
	fmt.Printf("loss %.7f\n", loss/4.0)
	// Output:
	// log(0.080) + log(0.501) -> -3.197 : -3.214
	// log(0.732) + log(0.679) -> -0.717 : -0.699
	// log(0.303) + log(0.584) -> -1.735 : -1.733
	// log(0.445) + log(0.350) -> -1.851 : -1.861
	// loss 0.0000014
}

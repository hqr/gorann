package gorann

import (
	"fmt"
	"math"
	"math/rand"
)

func ExampleF_xorbits() {
	rand.Seed(0) // for reproducible results
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"sigmoid", 16}
	output := NeuLayerConfig{"sigmoid", 1}
	nn := NewNeuNetwork(input, hidden, 2, output, ADAM)
	nn.tunables.alpha = 0.4
	// nn.tunables.batchsize = 10
	// nn.tunables.lambda = DEFAULT_lambda

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
		converged = nn.Train(Xs, TrainParams{resultvalcb: xorbits, repeat: 3, testingpct: 60, maxcost: 1E-3, maxbackprops: 1E7})
	}
	// test and print the results (expected output below)
	var crossen, mse float64
	for k := 0; k < 4; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		y1 := nn.Predict(xvec)
		y2 := xorbits(xvec)
		crossen += nn.CostCrossEntropy(y2)
		mse += nn.CostMse(y2)
		a, b, c, d := int(xvec[0]), int(xvec[1]), int(y1[0]), int(y2[0])
		fmt.Printf("%08b ^ %08b -> %08b : %08b\n", a, b, c, d)
	}
	fmt.Printf("cross-entropy %.5f, mse %.5f\n", crossen/4, mse/4)
	// Output:
	// 10010001 ^ 01101100 -> 11111001 : 11111101
	// 01101101 ^ 10011110 -> 11101111 : 11110011
	// 11001011 ^ 11011110 -> 00011000 : 00010101
	// 10111101 ^ 00000011 -> 10111101 : 10111110
	// cross-entropy 0.13692, mse 0.00006
}

func ExampleF_1() {
	rand.Seed(0) // for reproducible results
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"sigmoid", 4}
	output := NeuLayerConfig{"sigmoid", 1}
	nn := NewNeuNetwork(input, hidden, 2, output, ADAM)
	nn.tunables.alpha = 0.4
	nn.tunables.costfunction = CostCrossEntropy

	xorbits := func(xvec []float64) []float64 {
		var y []float64 = []float64{0}
		a := int(xvec[0])
		b := int(xvec[1])
		y[0] = float64(a ^ b)
		return y
	}
	Xs := newMatrix(100, 2)
	converged := 0
	for i := 0; i < len(Xs); i++ {
		Xs[i][0] = float64(rand.Int31n(2))
		Xs[i][1] = float64(rand.Int31n(2))
	}
	for converged == 0 {
		converged = nn.Train(Xs, TrainParams{resultvalcb: xorbits, testingpct: 90, maxcost: 1E-8, maxbackprops: 1E6})
	}
	// test and print the results (expected output below)
	var err, loss float64
	Xs = [][]float64{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}
	for _, xvec := range Xs {
		y1 := nn.Predict(xvec)
		y2 := xorbits(xvec)
		err += nn.AbsError(y2)
		loss += nn.CostCrossEntropy(y2)
		a, b, c, d := int(xvec[0]), int(xvec[1]), y1[0], int(y2[0])
		fmt.Printf("%01b ^ %01b -> %.1f : %01b\n", a, b, c, d)
	}
	// Output:
	// 0 ^ 0 -> 0.0 : 0
	// 0 ^ 1 -> 1.0 : 1
	// 1 ^ 0 -> 1.0 : 1
	// 1 ^ 1 -> 0.0 : 0
}

func ExampleF_sumsquares() {
	rand.Seed(0)
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"sigmoid", 16}
	output := NeuLayerConfig{"identity", 1}
	nn := NewNeuNetwork(input, hidden, 2, output, Rprop)

	sumsquares := func(xvec []float64) []float64 {
		var y []float64 = []float64{0}
		for i := 0; i < len(xvec); i++ {
			y[0] += xvec[i] * xvec[i]
		}
		return y
	}
	Xs := newMatrix(1000, 2)
	converged := 0
	for converged == 0 {
		for i := 0; i < len(Xs); i++ {
			Xs[i][0] = rand.Float64() / 1.5
			Xs[i][1] = rand.Float64() / 1.5
		}
		converged = nn.Train(Xs, TrainParams{resultvalcb: sumsquares, repeat: 3, testingpct: 30, maxcost: 5E-7})
	}
	// use to estimate
	var loss float64
	for i := 0; i < 4; i++ {
		var xvec []float64 = []float64{0, 0}
		xvec[0] = rand.Float64() / 1.5
		xvec[1] = rand.Float64() / 1.5
		y1 := nn.Predict(xvec)
		y2 := sumsquares(xvec)
		loss += nn.CostMse(y2)
		fmt.Printf("%.3f**2 + %.3f**2 -> %.3f : %.3f\n", xvec[0], xvec[1], y1[0], y2[0])
	}
	fmt.Printf("loss %.3e\n", loss/4.0)
	// Output:
	// 0.178**2 + 0.127**2 -> 0.049 : 0.048
	// 0.550**2 + 0.547**2 -> 0.604 : 0.602
	// 0.565**2 + 0.373**2 -> 0.459 : 0.458
	// 0.144**2 + 0.210**2 -> 0.066 : 0.065
	// loss 1.067e-06
}

func ExampleF_sumlogarithms() {
	rand.Seed(0)
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"tanh", 8}
	output := NeuLayerConfig{"tanh", 1}
	nn := NewNeuNetwork(input, hidden, 5, output, RMSprop)
	nn.tunables.momentum = 0.5
	nn.tunables.batchsize = 50

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
	for converged == 0 {
		for i := 0; i < len(Xs); i++ {
			Xs[i][0], Xs[i][1] = rand.Float64(), rand.Float64()
		}
		converged = nn.Train(Xs, TrainParams{resultvalcb: sumlogarithms, repeat: 3, maxcost: 1E-5, maxgradnorm: 0.01, maxbackprops: 4E6})
	}
	if converged&ConvergedMaxBackprops > 0 {
		fmt.Printf("maxed out on back propagations %d\n", nn.nbackprops)
	}
	var mse float64
	for i := 0; i < 4; i++ {
		var xvec []float64 = []float64{0, 0}
		xvec[0] = rand.Float64()
		xvec[1] = rand.Float64()
		y2 := sumlogarithms(xvec)
		y1 := nn.Predict(xvec)
		mse += nn.CostMse(y2)
		fmt.Printf("log(%.3f) + log(%.3f) -> %.3f : %.3f\n", xvec[0], xvec[1], y1[0], y2[0])
	}
	fmt.Printf("mse %.4e\n", mse/4)
	// Output:
	// log(0.250) + log(0.829) -> -1.589 : -1.574
	// log(0.637) + log(0.920) -> -0.575 : -0.535
	// log(0.840) + log(0.585) -> -0.745 : -0.711
	// log(0.571) + log(0.962) -> -0.638 : -0.599
	// mse 8.7610e-06
}

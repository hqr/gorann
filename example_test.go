package gorann

import (
	"fmt"
	"math"
	"math/rand"
)

func ExampleF_xorbits() {
	rand.Seed(0) // for reproducible results

	// create NN: input layer of 2 nodes, 3 hidden layers of 4 that utilize ReLU
	// and a single-node output layer that uses sigmoid() activation
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"sigmoid", 16}
	output := NeuLayerConfig{"sigmoid", 1}
	// tunables := NeuTunables{alpha: 0.4, momentum: 0.6, batchsize: BatchSGD, costfunction: CostLinear, gdalgname: RMSprop}
	tunables := NeuTunables{alpha: 0.4, momentum: 0.6, gdalgname: ADAM}
	nn := NewNeuNetwork(input, hidden, 2, output, tunables)
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
	for j := 0; j < 10000; j++ {
		// fill with random bits 0 to 0x1111111
		for i := 0; i < len(Xs); i++ {
			Xs[i][0] = float64(rand.Int31n(maxint))
			Xs[i][1] = float64(rand.Int31n(maxint))
		}
		// run each sample 2 times, use callback to compute true result for a given input
		for k := 0; k < 2; k++ {
			nn.Train(Xs, xorbits)
		}
		// debug
		if nn.tunables.tracking > 0 && j%1000 == 0 {
			nn.printTracks(j)
		}
	}
	// test and print the results (expected output below)
	var err, loss float64
	for i := 0; i < 4; i++ {
		var xvec []float64 = []float64{0, 0}
		xvec[0] = float64(rand.Int31n(maxint))
		xvec[1] = float64(rand.Int31n(maxint))
		y1 := nn.Predict(xvec)
		y2 := xorbits(xvec)
		err += nn.AbsError(y2)
		loss += nn.CostLinear(y2)

		a := int(xvec[0])
		b := int(xvec[1])
		c := int(y1[0])
		d := int(y2[0])
		fmt.Printf("%08b ^ %08b -> %08b : %08b\n", a, b, c, d)
	}
	fmt.Printf("error %d, loss %.5f\n", int(err)/4, loss/4)
	// Output:
	// 01110101 ^ 10001001 -> 11111101 : 11111100
	// 01000010 ^ 01011100 -> 01000011 : 00011110
	// 10101111 ^ 01110101 -> 11010100 : 11011010
	// 10110011 ^ 11101100 -> 01011010 : 01011111
	// error 12, loss 0.00279
}

func ExampleF_sumsquares() {
	rand.Seed(0)
	// NN: input layer of 2 nodes, 2 hidden layers consisting of 6 nodes that utilize tanh(),
	// and a single-node output layer that uses sigmoid() activation
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"tanh", 6}
	output := NeuLayerConfig{"sigmoid", 1}
	tunables := NeuTunables{alpha: 0.01, momentum: 0.999, batchsize: 10, gdalgname: Adadelta}
	nn := NewNeuNetwork(input, hidden, 2, output, tunables)

	sumsquares := func(xvec []float64) []float64 {
		var y []float64 = []float64{0}
		for i := 0; i < len(xvec); i++ {
			y[0] += xvec[i] * xvec[i]
		}
		return y
	}
	Xs := newMatrix(100, 2)
	for j := 0; j < 1000; j++ {
		// fill with random [0, 1.0)
		for i := 0; i < len(Xs); i++ {
			Xs[i][0] = rand.Float64() / 1.5
			Xs[i][1] = rand.Float64() / 1.5
		}
		// train the network: run each sample 2 times and use the callback to compute true result
		for k := 0; k < 2; k++ {
			nn.Train(Xs, sumsquares)
		}
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
	fmt.Printf("loss %.5f\n", loss/4.0)
	// Output:
	// 0.636**2 + 0.501**2 -> 0.666 : 0.656
	// 0.132**2 + 0.453**2 -> 0.228 : 0.222
	// 0.660**2 + 0.366**2 -> 0.568 : 0.570
	// 0.425**2 + 0.638**2 -> 0.595 : 0.588
	// loss 0.00002
}

func ExampleF_sumlogarithms() {
	rand.Seed(1)
	// create NN: input layer of 2 nodes, 2 hidden layers consisting of 6 nodes that utilize tanh(),
	// and a single-node output layer that uses sigmoid() activation
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"tanh", 8}
	output := NeuLayerConfig{"tanh", 1}
	tunables := NeuTunables{alpha: 0.01, momentum: 0.5, batchsize: 10, gdalgname: RMSprop} //, gdalgscope: 1}
	nn := NewNeuNetwork(input, hidden, 4, output, tunables)

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

	for j := 0; j < 1000; j++ {
		// run each sample 2 times, use callback to compute true result for a given input
		for i := 0; i < len(Xs); i++ {
			Xs[i][0] = rand.Float64()
			Xs[i][1] = rand.Float64()
		}
		for k := 0; k < 2; k++ {
			nn.Train(Xs, sumlogarithms)
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
	fmt.Printf("loss %.5f\n", loss/4.0)
	// Output:
	// log(0.925) + log(0.139) -> -1.977 : -2.052
	// log(0.690) + log(0.329) -> -1.502 : -1.482
	// log(0.375) + log(0.214) -> -2.428 : -2.522
	// log(0.569) + log(0.246) -> -1.945 : -1.968
	// loss 0.00003
}

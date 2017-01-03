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
	tunables := NeuTunables{alpha: 0.1, momentum: 0.3, batchsize: BatchSGD, costfunction: CostLinear, gdalgname: RMSprop}
	nn := NewNeuNetwork(input, hidden, 2, output, tunables)

	normalize := func(vec []float64) {
		divElemVector(vec, float64(math.MaxInt8))
	}
	denormalize := func(vec []float64) {
		mulElemVector(vec, float64(math.MaxInt8))
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
	for j := 0; j < 1000; j++ {
		// fill with random bits 0 to 0x1111111
		for i := 0; i < len(Xs); i++ {
			Xs[i][0] = float64(rand.Int31n(math.MaxInt8))
			Xs[i][1] = float64(rand.Int31n(math.MaxInt8))
		}
		// run each sample 2 times, use callback to compute true result for a given input
		for k := 0; k < 2; k++ {
			nn.Train(Xs, xorbits)
		}
	}
	// test and print the results (expected output below)
	var loss float64
	for i := 0; i < 4; i++ {
		var xvec []float64 = []float64{0, 0}
		xvec[0] = float64(rand.Int31n(math.MaxInt8))
		xvec[1] = float64(rand.Int31n(math.MaxInt8))
		y1 := nn.Predict(xvec)
		y2 := xorbits(xvec)
		loss += nn.CostLinear(y2)

		a := int(xvec[0])
		b := int(xvec[1])
		c := int(y1[0])
		d := int(y2[0])
		fmt.Printf("%07b XOR %07b -> %07b : %07b\n", a, b, c, d)
	}
	fmt.Printf("loss %.5f\n", loss/4.0)
	// Output:
	// 0000011 XOR 1011010 -> 1010101 : 1011001
	// 0010001 XOR 0001101 -> 0001001 : 0011100
	// 0111011 XOR 1011010 -> 1011011 : 1100001
	// 1001010 XOR 1110000 -> 0111110 : 0111010
	// loss 0.00321
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

	sumlogarithms := func(xvec []float64) []float64 {
		var y []float64 = []float64{0}
		for i := 0; i < len(xvec); i++ {
			y[0] += math.Log(xvec[i])
		}
		y[0] = -y[0] / 8
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
		y1 := nn.Predict(xvec)
		y2 := sumlogarithms(xvec)
		loss += nn.CostLinear(y2)
		fmt.Printf("log(%.3f) + log(%.3f) -> %.3f : %.3f\n", xvec[0], xvec[1], y1[0], y2[0])
	}
	fmt.Printf("loss %.5f\n", loss/4.0)
	// Output:
	// log(0.925) + log(0.139) -> 0.249 : 0.256
	// log(0.690) + log(0.329) -> 0.187 : 0.185
	// log(0.375) + log(0.214) -> 0.300 : 0.315
	// log(0.569) + log(0.246) -> 0.241 : 0.246
	// loss 0.00004
}

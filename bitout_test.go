package gorann

import (
	"fmt"
	"math/rand"
	"testing"
)

func Test_bitoutput(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 16}
	hidden := NeuLayerConfig{"sigmoid", 16}
	output := NeuLayerConfig{"sigmoid", 8}
	nn := NewNeuNetwork(input, hidden, 3, output, &NeuTunables{gdalgname: RMSprop})
	nn.initXavier()

	xor8bits := func(xvec []float64) []float64 {
		var y = make([]float64, 8)
		for i := 0; i < 8; i++ {
			y[i] = float64(int(xvec[i]) ^ int(xvec[8+i]))
		}
		return y
	}
	Xs := newMatrix(1000, 16)
	for i := 0; i < len(Xs); i++ {
		const maxint = int32(0xffff)
		x := rand.Int31n(maxint)
		for j := 0; j < 16; j++ {
			Xs[i][j] = float64(x & 1)
			x >>= 1
		}
	}
	converged := 0
	ttp := &TTP{nn: nn, resultvalcb: xor8bits, pct: 50, maxcost: 0.3, maxbackprops: 1E6}
	for converged == 0 {
		converged = nn.Train(Xs, ttp)
	}
	if converged&ConvergedMaxBackprops > 0 {
		t.Errorf("reached the maximum number of back propagations (%d)\n", nn.nbackprops)
	}
	var crossen, mse float64
	for k := 0; k < 4; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		yvec := xor8bits(xvec)
		avec := nn.Predict(xvec)
		crossen += nn.CostCrossEntropy(yvec)
		mse += nn.CostMse(yvec)
		for i := 0; i < 8; i++ {
			if avec[i] < 0.4 {
				avec[i] = 0
			} else if avec[i] > 0.6 {
				avec[i] = 1
			}
		}
		fmt.Printf("%v ^ %v -> %.1v : %v\n", xvec[:8], xvec[8:], avec, yvec)
	}
	fmt.Printf("cross-entropy %.5f, mse %.5f\n", crossen/4, mse/4)
}

func Test_classify(t *testing.T) {
	rand.Seed(0)
	nclasses := 3
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"sigmoid", 20}
	output := NeuLayerConfig{"softmax", nclasses}
	nn := NewNeuNetwork(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, costfname: CostCrossEntropy, batchsize: 10})
	maxint := int32(0xff)
	normalize := func(vec []float64) {
		divElemVector(vec, float64(maxint))
	}
	nn.callbacks = NeuCallbacks{normalize, nil, nil}

	xorclassify := func(xvec []float64) []float64 {
		var y = make([]float64, nclasses)
		i := (int(xvec[0]) ^ int(xvec[1])) % nclasses
		y[nclasses-i-1] = 1
		return y
	}
	Xs := newMatrix(200, 2)
	ttp := &TTP{nn: nn, resultvalcb: xorclassify, pct: 50, maxcost: 0.4, maxbackprops: 5E6}
	for i := 0; i < len(Xs); i++ {
		Xs[i][0] = float64(rand.Int31n(maxint))
		Xs[i][1] = float64(rand.Int31n(maxint))
	}
	converged := 0
	for converged == 0 {
		converged = nn.Train(Xs, ttp)
	}
	var crossen, mse float64
	for k := 0; k < 8; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		k := (int(xvec[0]) ^ int(xvec[1])) % nclasses
		yvec := xorclassify(xvec)
		avec := nn.Predict(xvec)
		crossen += nn.CostCrossEntropy(yvec)
		mse += nn.CostMse(yvec)
		for i := 0; i < nclasses; i++ {
			if avec[i] < 0.4 {
				avec[i] = 0
			} else if avec[i] > 0.6 {
				avec[i] = 1
			}
		}
		fmt.Printf("%3v ^ %3v (%1d) -> %.1v : %v\n", xvec[0], xvec[1], k, avec, yvec)
	}
	fmt.Printf("cross-entropy %.5f, mse %.5f\n", crossen/4, mse/4)
	if converged&ConvergedMaxBackprops > 0 {
		t.Errorf("reached the maximum number of back propagations (%d)\n", nn.nbackprops)
	}
}

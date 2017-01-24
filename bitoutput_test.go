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
	nn := NewNeuNetwork(input, hidden, 3, output, RMSprop)
	nn.initXavier()

	xor8bits := func(xvec []float64) []float64 {
		var y []float64 = make([]float64, 8)
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
	for converged == 0 {
		converged = nn.Train(Xs, TrainParams{resultvalcb: xor8bits, testingpct: 50, maxcost: 0.3, maxbackprops: 1E6})
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

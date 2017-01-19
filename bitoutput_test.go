package gorann

import (
	"fmt"
	"math/rand"
	"testing"
)

func Test_bitoutput(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"tanh", 8}
	output := NeuLayerConfig{"sigmoid", 8}
	nn := NewNeuNetwork(input, hidden, 2, output, RMSprop)
	nn.tunables.costfunction = CostMse

	maxint := int32(255)
	normalize := func(vec []float64) {
		divElemVector(vec, float64(maxint))
	}
	nn.callbacks = NeuCallbacks{normcbX: normalize}
	xor8bits := func(xvec []float64) []float64 {
		var y []float64 = make([]float64, 8)
		result := int(xvec[0]) ^ int(xvec[1])
		mask := 1
		for i := 0; i < 8; i++ {
			if result&mask > 0 {
				y[7-i] = 1
			}
			mask <<= 1
		}
		return y
	}
	Xs := newMatrix(16, 2)
	for i := 0; i < len(Xs); i++ {
		Xs[i][0] = float64(rand.Int31n(maxint))
		Xs[i][1] = float64(rand.Int31n(maxint))
	}
	converged := 0
	for converged == 0 {
		converged = nn.Train(Xs, TrainParams{resultvalcb: xor8bits, testingpct: 50, maxcost: 1E-1, maxbackprops: 1E6})
	}
	if converged&ConvergedMaxBackprops > 0 {
		t.Errorf("reached the maximum number of back propagations (%d)\n", nn.nbackprops)
	}
	var loss float64
	for k := 0; k < 4; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		y2 := xor8bits(xvec)
		y1 := nn.Predict(xvec)
		for j := 0; j < len(y1); j++ {
			loss += nn.CostMse(y2) // CostCrossEntropy
			if y1[j] < 0.3 {
				y1[j] = 0
			} else if y1[j] > 0.7 {
				y1[j] = 1
			}
		}
		fmt.Printf("%08b ^ %08b -> %.1v : %v\n", int(xvec[0]), int(xvec[1]), y1, y2)
	}
	fmt.Println("loss", loss/4)
}

package gorann

import (
	"fmt"
	"math/rand"
	"testing"
)

func Test_bitoutput(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"sigmoid", 16}
	output := NeuLayerConfig{"sigmoid", 8}
	nn := NewNeuNetwork(input, hidden, 2, output, RMSprop)
	// nn.tunables.costfunction = CostLogistic
	// nn.layers[2].config.actfname = "leakyrelu"
	nn.tunables.gdalgscopeall = true

	maxint := int32(256)
	normalize := func(vec []float64) {
		divElemVector(vec, float64(maxint))
	}
	nn.callbacks = NeuCallbacks{normcbX: normalize}
	xor8bits := func(xvec []float64) []float64 {
		var y []float64 = make([]float64, 8)
		result := int(xvec[0]) ^ int(xvec[1])
		bit := 1
		for i := 0; i < 8; i++ {
			if result&bit > 0 {
				y[7-i] = 1
			}
			bit <<= 1
		}
		return y
	}
	Xs := newMatrix(1000, 2)
	converged := 0
	for converged == 0 {
		for i := 0; i < len(Xs); i++ {
			Xs[i][0] = float64(rand.Int31n(maxint))
			Xs[i][1] = float64(rand.Int31n(maxint))
		}
		converged = nn.Train(Xs, TrainParams{resultvalcb: xor8bits, repeat: 3, testingpct: 10, maxcost: 0.8, maxbackprops: 1E6})
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
			loss += nn.CostLinear(y2)
			// loss += y2[j]*math.Log(y1[j]) + (1-y2[j])*math.Log(1-y1[j])
			if y1[j] < 0.4 {
				y1[j] = 0
			} else if y1[j] > 0.6 {
				y1[j] = 1
			}
		}
		fmt.Printf("%08b ^ %08b -> %.2v : %v\n", int(xvec[0]), int(xvec[1]), y1, y2)
	}
	fmt.Println("loss", loss/4)
}

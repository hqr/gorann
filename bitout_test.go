package gorann

import (
	"fmt"
	"math/rand"
	"testing"
)

func Test_axor8bits(t *testing.T) {
	rand.Seed(0)
	xor8bits(t, nnA8bits())
}

func Test_bxor8bits(t *testing.T) {
	rand.Seed(0)
	xor8bits(t, nnB8bits())
}

func Test_cxor8bits(t *testing.T) {
	rand.Seed(0)
	xor8bits(t, nnC8bits())
}

func Test_mixor8bits(t *testing.T) {
	rand.Seed(0)
	// mixer := NewWeightedMixerNN(nnA8bits(), nnB8bits(), nnC8bits())
	mixer := NewWeightedMixerNN(nnB8bits(), nnA8bits())
	nn := &mixer.NeuNetwork
	xor8bits(t, nn)

	for i := 0; i < len(mixer.nns); i++ {
		osize := mixer.olayer.size
		ii := i * osize
		fmt.Printf("%.2v\n", mixer.weights[ii:ii+osize])
	}
}

// common parts for the 3 tests above
func nnA8bits() *NeuNetwork {
	input := NeuLayerConfig{size: 16}
	hidden := NeuLayerConfig{"sigmoid", 16}
	output := NeuLayerConfig{"sigmoid", 8}
	nn := NewNeuNetwork(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM})
	nn.initXavier()
	return nn
}

func nnB8bits() *NeuNetwork {
	input := NeuLayerConfig{size: 16}
	hidden := NeuLayerConfig{"sigmoid", 32}
	output := NeuLayerConfig{"sigmoid", 8}
	nn := NewNeuNetwork(input, hidden, 2, output, &NeuTunables{gdalgname: RMSprop})
	nn.initXavier()
	return nn
}

func nnC8bits() *NeuNetwork {
	input := NeuLayerConfig{size: 16}
	hidden := NeuLayerConfig{"sigmoid", 48}
	output := NeuLayerConfig{"sigmoid", 8}
	nn := NewNeuNetwork(input, hidden, 1, output, &NeuTunables{gdalgname: RMSprop})
	nn.initXavier()
	return nn
}

func xor8bits(t *testing.T, nn *NeuNetwork) {
	_xor8bits := func(xvec []float64) []float64 {
		var y = make([]float64, 8)
		for i := 0; i < 8; i++ {
			y[i] = float64(int(xvec[i]) ^ int(xvec[8+i]))
		}
		return y
	}
	Xs := newMatrix(10000, 16)
	const maxint = int32(0xffff)
	// train data
	fillrandom := func() {
		for i := 0; i < len(Xs); i++ {
			x := rand.Int31n(maxint)
			for j := 0; j < 16; j++ {
				Xs[i][j] = float64(x & 1)
				x >>= 1
			}
		}
	}
	fillrandom()
	converged := 0
	ttp := &TTP{nn: nn, resultvalcb: _xor8bits, pct: 50, maxbackprops: 5E5}
	for converged == 0 {
		converged = nn.Train(Xs, ttp)
	}
	var crossen, mse float64
	// try the trained network on some new data
	fillrandom()
	for k := 0; k < 8; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		yvec := _xor8bits(xvec)
		avec := nn.Predict(xvec)
		crossen += nn.CostCrossEntropy(yvec)
		mse += nn.CostMse(yvec)
		mark := ""
		for i := 0; i < 8; i++ {
			if avec[i] < 0.4 {
				avec[i] = 0
			} else if avec[i] > 0.6 {
				avec[i] = 1
			}
			if avec[i] != yvec[i] {
				mark = "*"
			}
		}
		fmt.Printf("%v ^ %v -> %.1v : %v%s\n", xvec[:8], xvec[8:], avec, yvec, mark)
	}
	fmt.Printf("cross-entropy %.5f, mse %.5f\n", crossen/4, mse/4)
}

//==========================
// test cross entropy cost
//==========================
func Test_classify(t *testing.T) {
	rand.Seed(0)
	nclasses := 3
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"sigmoid", 20}
	output := NeuLayerConfig{"softmax", nclasses}
	nn := NewNeuNetwork(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, costfname: CostCrossEntropy, batchsize: 10})
	maxint := int32(0xff)
	normalize := func(vec []float64) {
		divVectorNum(vec, float64(maxint))
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

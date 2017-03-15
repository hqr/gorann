package gorann

import (
	"fmt"
	"hash/crc32"
	"math"
	"math/rand"
	"testing"
)

// const c_numInputBits = 32
const c_numInputBits = 16

func Test_axorn(t *testing.T) {
	rand.Seed(0)
	xornbits(t, nnAnbits())
}

func Test_bxorn(t *testing.T) {
	rand.Seed(0)
	xornbits(t, nnBnbits())
}

func Test_cxorn(t *testing.T) {
	rand.Seed(0)
	xornbits(t, nnCnbits())
}

func Test_mixorn(t *testing.T) {
	rand.Seed(0)
	mixer := NewWeightedMixerNN(nnAnbits(), nnBnbits(), nnCnbits())
	nn := &mixer.NeuNetwork
	xornbits(t, nn)
}

func Test_gradxorn(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: c_numInputBits}
	output := NeuLayerConfig{"sigmoid", input.size / 2}
	nns_i := make([]NeuNetworkInterface, 16)
	nns := nns_i[:0]
	for ksize := 1; ksize < 4; ksize++ {
		for knum := 1; knum < 4; knum++ {
			hidden := NeuLayerConfig{"sigmoid", input.size * ksize}
			nn := NewNeuNetwork(input, hidden, knum, output, &NeuTunables{gdalgname: ADAM, batchsize: 10, winit: Xavier})
			nns = append(nns, nn)
		}
	}
	mixer := NewWeightedGradientNN(nns...)
	nn := &mixer.NeuNetwork
	xornbits(t, nn)
}

// common parts for the 3 tests above
func nnAnbits() *NeuNetwork {
	input := NeuLayerConfig{size: c_numInputBits}
	hidden := NeuLayerConfig{"sigmoid", input.size}
	output := NeuLayerConfig{"sigmoid", input.size / 2}
	nn := NewNeuNetwork(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, batchsize: 10, winit: Xavier})
	return nn
}

func nnBnbits() *NeuNetwork {
	input := NeuLayerConfig{size: c_numInputBits}
	hidden := NeuLayerConfig{"sigmoid", input.size * 2}
	output := NeuLayerConfig{"sigmoid", input.size / 2}
	nn := NewNeuNetwork(input, hidden, 2, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: Xavier})
	return nn
}

func nnCnbits() *NeuNetwork {
	input := NeuLayerConfig{size: c_numInputBits}
	hidden := NeuLayerConfig{"sigmoid", input.size * 3}
	output := NeuLayerConfig{"sigmoid", input.size / 2}
	nn := NewNeuNetwork(input, hidden, 1, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: Xavier})
	return nn
}

func xornbits(t *testing.T, nn *NeuNetwork) {
	osize := nn.getCoutput().size
	_xornbits := func(xvec []float64) []float64 {
		var y = make([]float64, osize)
		for i := 0; i < osize; i++ {
			y[i] = float64(int(xvec[i]) ^ int(xvec[osize+i]))
		}
		return y
	}
	Xs := newMatrix(10000, nn.nnint.getIsize())
	// train data
	fillrandom := func() {
		for i := 0; i < len(Xs); i++ {
			x := rand.Int63n(math.MaxInt64)
			for j := 0; j < nn.nnint.getIsize(); j++ {
				Xs[i][j] = float64(x & 1)
				x >>= 1
				if x == 0 {
					x = rand.Int63n(math.MaxInt64)
				}
			}
		}
	}
	fillrandom()
	converged := 0
	ttp := &TTP{nn: nn, resultvalcb: _xornbits, pct: 50, maxbackprops: 2E6}
	for converged == 0 {
		converged = nn.Train(Xs, ttp)
	}
	var crossen, mse float64
	// try the trained network on some new data
	fillrandom()
	num := 8
	for k := 0; k < num; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		yvec := _xornbits(xvec)
		avec := nn.Predict(xvec)
		crossen += nn.CostCrossEntropy(yvec)
		mse += nn.CostMse(yvec)
		mark := ""
		for i := 0; i < len(avec); i++ {
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
	fmt.Printf("cross-entropy %.5f, mse %.5f\n", crossen/float64(num), mse/float64(num))
}

//================================================================================
//
// TRANSFORM --- TRANSFORM --- TRANSFORM --- TRANSFORM
// 16 bits (bit per NN input) => log() transform => 16 bits (16 output neurons)
//
//================================================================================

func Test_transform_56_2(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 16}
	hidden := NeuLayerConfig{"sigmoid", 56}
	output := NeuLayerConfig{"sigmoid", 16}
	nn := NewNeuNetwork(input, hidden, 2, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: Xavier})
	transformBits(t, nn)
}

func Test_transform_48_2(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 16}
	hidden := NeuLayerConfig{"sigmoid", 48}
	output := NeuLayerConfig{"sigmoid", 16}
	nn := NewNeuNetwork(input, hidden, 2, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: Xavier})
	transformBits(t, nn)
}

func Test_transform_32_2(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 16}
	hidden := NeuLayerConfig{"sigmoid", 32}
	output := NeuLayerConfig{"sigmoid", 16}
	nn := NewNeuNetwork(input, hidden, 2, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: Xavier})
	transformBits(t, nn)
}

func Test_transform_40_1(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 16}
	hidden := NeuLayerConfig{"sigmoid", 40}
	output := NeuLayerConfig{"sigmoid", 16}
	nn := NewNeuNetwork(input, hidden, 1, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: Xavier})
	transformBits(t, nn)
}

func Test_transform_48_1(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 16}
	hidden := NeuLayerConfig{"sigmoid", 48}
	output := NeuLayerConfig{"sigmoid", 16}
	nn := NewNeuNetwork(input, hidden, 1, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: Xavier})
	transformBits(t, nn)
}

func Test_mixtransform(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 16}
	output := NeuLayerConfig{"sigmoid", 16}
	nns_i := make([]NeuNetworkInterface, 32)
	nns := nns_i[:0]
	for ksize := 0; ksize < 6; ksize++ {
		hsize := input.size + ksize*input.size/2
		for knum := 1; knum < 4; knum++ {
			hidden := NeuLayerConfig{"sigmoid", hsize}
			nn := NewNeuNetwork(input, hidden, knum, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: Xavier})
			nns = append(nns, nn)
		}
	}
	mixer := NewWeightedGradientNN(nns...)
	nn := &mixer.NeuNetwork
	transformBits(t, nn)
}

//
// 16 x input --> log() --> 16 x output
//
func transformBits(t *testing.T, nn *NeuNetwork) {
	isize, osize := 16, 16
	_transformBits := func(xvec []float64) []float64 {
		var y = make([]float64, osize)
		x, bit, fx := 0, 1, 0.0
		for j := 0; j < isize; j++ {
			if xvec[j] == 1 {
				x += bit
			}
			bit <<= 1
		}
		fx = math.Log(float64(x)) * 1000
		fy := int(fx)
		for j := 0; j < osize; j++ {
			y[j] = float64(fy & 1)
			fy >>= 1
		}
		return y
	}
	Xs := newMatrix(10000, nn.nnint.getIsize())
	// train data
	fillrandom := func() {
		for i := 0; i < len(Xs); i++ {
			x := rand.Intn(65535)
			for j := 0; j < isize; j++ {
				Xs[i][j] = float64(x & 1)
				x >>= 1
			}
		}
	}
	fillrandom()
	converged := 0
	ttp := &TTP{nn: nn, resultvalcb: _transformBits, pct: 50, maxbackprops: 1E7}
	for converged == 0 {
		converged = nn.Train(Xs, ttp)
	}
	var crossen, mse float64
	// try the trained network on some new data
	fillrandom()
	num := 8
	for k := 0; k < num; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		yvec := _transformBits(xvec)
		avec := nn.Predict(xvec)
		crossen += nn.CostCrossEntropy(yvec)
		mse += nn.CostMse(yvec)
		mark := ""
		for i := 0; i < len(avec); i++ {
			if avec[i] < 0.4 {
				avec[i] = 0
			} else if avec[i] > 0.6 {
				avec[i] = 1
			}
			if avec[i] != yvec[i] {
				mark = "*"
			}
		}
		fmt.Printf("log(%v) -> %.1v : %v%s\n", xvec, avec, yvec, mark)
	}
	fmt.Printf("cross-entropy %.5f, mse %.5f\n", crossen/float64(num), mse/float64(num))
}

//============================================================================
//
// cross entropy cost
//
//============================================================================
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
	nn.callbacks = &NeuCallbacks{normalize, nil, nil}

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
	num := 8
	for k := 0; k < num; k++ {
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
	fmt.Printf("cross-entropy %.5f, mse %.5f\n", crossen/float64(num), mse/float64(num))
	if converged&ConvergedMaxBackprops > 0 {
		t.Errorf("reached the maximum number of back propagations (%d)\n", nn.nbackprops)
	}
}

//=================================================================================
//
//
//
//=================================================================================
// s/test/Test/ to enable
func test_acrc32c(t *testing.T) {
	rand.Seed(0)
	crc32cbytes(t, nnAcrc32cbytes())
}

func nnAcrc32cbytes() *NeuNetwork {
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"tanh", input.size * 32}
	output := NeuLayerConfig{"sigmoid", 1}
	nn := NewNeuNetwork(input, hidden, 4, output, &NeuTunables{gdalgname: ADAM, winit: Xavier})
	return nn
}

func crc32cbytes(t *testing.T, nn *NeuNetwork) {
	crc32Ctab := crc32.MakeTable(crc32.Castagnoli)

	_crc32cbytes := func(xvec []float64) []float64 {
		var bytes = make([]byte, nn.nnint.getIsize())
		for j := 0; j < nn.nnint.getIsize(); j++ {
			bytes[j] = byte(xvec[j] * 256)
		}

		crc := crc32.Checksum(bytes, crc32Ctab)
		y := float64(crc) / float64(math.MaxUint32)
		return []float64{y}
	}

	Xs := newMatrix(10000, nn.nnint.getIsize())
	// train data
	fillrandombytes := func() {
		for i := 0; i < len(Xs); i++ {
			for j := 0; j < nn.nnint.getIsize(); j++ {
				Xs[i][j] = float64(rand.Int31n(256)) / float64(256)
			}
		}
	}
	fillrandombytes()
	converged := 0
	ttp := &TTP{nn: nn, resultvalcb: _crc32cbytes, pct: 50, maxbackprops: 1E7}
	for converged == 0 {
		converged = nn.Train(Xs, ttp)
	}
	var mse float64
	// try the trained network on some new data
	fillrandombytes()
	num := 8
	for k := 0; k < num; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		var bytes = make([]byte, nn.nnint.getIsize())
		for j := 0; j < nn.nnint.getIsize(); j++ {
			bytes[j] = byte(xvec[j] * 256)
		}
		yvec := _crc32cbytes(xvec)
		avec := nn.Predict(xvec)
		mse += nn.CostMse(yvec)
		fmt.Printf("crc32c(%3v) -> %v : %v\n", bytes, uint32(avec[0]*math.MaxUint32), uint32(yvec[0]*math.MaxUint32))
	}
	fmt.Printf("mse %.5f\n", mse/float64(num))
}

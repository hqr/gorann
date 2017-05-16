package gorann

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

//==================================================================================
//
// Evolution: the most basic test
// see also: http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
//
//==================================================================================
var etu = &EvoTunables{}
var constant = []float64{math.Pi / 5, -math.E / 3, math.Phi / 2} // normalized
var _constant = func(xvec []float64) []float64 { return constant }

func Test_constant(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 3}
	hidden := NeuLayerConfig{"tanh", 9}
	output := NeuLayerConfig{"identity", 3}

	// (un)comment, to use either GD or Evolution
	// tu := &NeuTunables{gdalgname: ADAM, batchsize: 10, winit: Xavier}
	// nn := NewNeuNetwork(input, hidden, 2, output, tu)
	tu := &NeuTunables{batchsize: 10, winit: Xavier}
	etu = &EvoTunables{
		NeuTunables: *tu,
		sigma:       0.1,   // normal-noise(0, sigma)
		momentum:    0.2,   //
		hireward:    5.0,   // diff (in num stds) that warrants a higher learning rate
		hialpha:     0.1,   //
		rewd:        0.001, //
		rewdup:      10,    // reward delta doubling period
		nperturb:    64,    // half the number of the NN weight fluctuations aka jitters
		sparsity:    0,     // noise matrix sparsity (%)
		jinflate:    4}     // gaussian noise inflation ratio, to speed up evolution
	nn := NewEvolution(input, hidden, 3, output, etu, 1)

	// test
	Xs := newMatrix(1000, 3, 0.0, 1.0)
	ttp := &TTP{nn: nn, resultvalcb: _constant, maxcost: 1E-4, maxbackprops: 1E4}
	for cnv := 0; cnv == 0; {
		cnv = ttp.Train(TtpArr(Xs))
	}
	var mse float64
	Xs = newMatrix(8, 3, 0.0, 1.0)
	for i := 0; i < 8; i++ {
		y2 := _constant(Xs[i])
		y1 := nn.Predict(Xs[i])
		mse += nn.costfunction(y2)
		fmt.Printf("%.3f : %.3f\n", y1, y2)
	}
	fmt.Printf("mse %.3e\n", mse/8)
}

//==================================================================================
//
// https://rmcantin.bitbucket.io/html/demos.html#hart6func
//
//==================================================================================
var hart6_mA = [][]float64{
	{10.0, 3.0, 17.0, 3.5, 1.7, 8.0},
	{0.05, 10.0, 17.0, 0.1, 8.0, 14.0},
	{3.0, 3.5, 1.7, 10.0, 17.0, 8.0},
	{17.0, 8.0, 0.05, 10.0, 0.1, 14.0}}

var hart6_mC = []float64{1.0, 1.2, 3.0, 3.2}

var hart6_mP = [][]float64{
	{0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886},
	{0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991},
	{0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650},
	{0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381}}

var hart6_opt = []float64{0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573}

var hart6_nn *NeuNetwork
var hart6_newrand *rand.Rand
var hart6_cpy = []float64{0, 0, 0, 0, 0, 0}

func fn_hartmann(xvec []float64) (yvec []float64) {
	yvec = []float64{0}
	for i := 0; i < 4; i++ {
		sum := 0.0
		for j := 0; j < 6; j++ {
			val := xvec[j] - hart6_mP[i][j]
			sum -= hart6_mA[i][j] * val * val
		}
		yvec[0] -= hart6_mC[i] * math.Exp(sum)
	}
	yvec[0] /= -3.33 // NOTE: normalize into (0, 1)
	return
}

func hart6_fill(Xs [][]float64) {
	for i := 0; i < len(Xs); i++ {
		prevmax := -1.0
		for j := 0; j < 10; j++ {
			fillVector(hart6_cpy, 0.0, 1.0, "", hart6_newrand)
			y := hart6_nn.nnint.forward(hart6_cpy)
			if y[0] > prevmax {
				copy(Xs[i], hart6_cpy)
				prevmax = y[0]
			}
		}
	}
}

func Test_hartmann(t *testing.T) {
	input := NeuLayerConfig{size: 6}
	hidden := NeuLayerConfig{"tanh", input.size * 3}
	output := NeuLayerConfig{"sigmoid", 1}
	tu := &NeuTunables{gdalgname: ADAM, batchsize: 10, winit: XavierNewRand}
	etu := &EvoTunables{
		NeuTunables: *tu,
		sigma:       0.1,           // normal-noise(0, sigma)
		momentum:    0.01,          //
		hireward:    5.0,           // diff (in num stds) that warrants a higher learning rate
		hialpha:     0.01,          //
		rewd:        0.001,         // FIXME: consider using reward / 1000
		rewdup:      rclr.coperiod, // reward delta doubling period
		nperturb:    64,            // half the number of the NN weight fluctuations aka jitters
		sparsity:    10,            // noise matrix sparsity (%)
		jinflate:    2}             // gaussian noise inflation ratio, to speed up evolution

	evo := NewEvolution(input, hidden, 3, output, etu, 0)
	nn := &evo.NeuNetwork
	hart6_nn = &evo.NeuNetwork
	hart6_newrand = evo.newrand
	evo.initXavier(hart6_newrand)

	evo.nnint = nn // NOTE: comment this statement, to compare versus Evolution

	//
	// henceforth, the usual training (regression) on random samples
	//
	Xs := newMatrix(10000, evo.getIsize())
	prevmax := 0.0
	ttp := &TTP{nn: evo, resultvalcb: fn_hartmann, pct: 10, maxbackprops: 1E7, repeat: 3}
	for cnv := 0; cnv == 0; {
		hart6_fill(Xs)
		cnv = ttp.Train(TtpArr(Xs))

		for i := 0; i < len(Xs); i++ {
			avec := nn.nnint.forward(hart6_opt)
			if avec[0] > prevmax {
				fmt.Printf("%.5f : %.5f\n", hart6_opt, avec[0])
				prevmax = avec[0]
			}
		}
	}
	var mse float64
	// try the trained network on some new data
	hart6_fill(Xs)
	num := 8
	for k := 0; k < num; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		yvec := fn_hartmann(xvec)
		nn.nnint.forward(xvec)
		mse += nn.costfunction(yvec)
	}
	fmt.Printf("mse %.5f\n", mse/float64(num))
}

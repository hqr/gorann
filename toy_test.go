package gorann

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
)

//==================================================================================
//
// Evolution: the most basic test
// see also: http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
//
//==================================================================================
var c_etu = &EvoTunables{
	sigma:    0.1,   // normal-noise(0, sigma)
	momentum: 0.2,   //
	hireward: 5.0,   // diff (in num stds) that warrants a higher learning rate
	hialpha:  0.1,   //
	rewd:     0.001, //
	rewdup:   10,    // reward delta doubling period
	nperturb: 64,    // half the number of the NN weight fluctuations aka jitters
	sparsity: 0,     // noise matrix sparsity (%)
	jinflate: 4,     // gaussian noise inflation ratio, to speed up evolution
}
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
	tu := &NeuTunables{batchsize: 6, winit: Xavier}
	etu := &EvoTunables{}
	copyStruct(etu, c_etu)
	copyStruct(&etu.NeuTunables, tu)
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

var hart6_newrand *rand.Rand

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

//=====================================================================
//
// Looking for maximum(fn_hartmann) - in PARALLEL
// -----------------   EXPERIMENTAL  ------------------------
//
//=====================================================================
type hart6_NN struct {
	NeuNetworkInterface
	xmax []float64
	xsgn []float64
	xprv []float64
	ymax float64
	yprv float64
	cost float64
	cprv float64
}

type ByYmax []*NeuRunner

func (rvec ByYmax) Len() int      { return len(rvec) }
func (rvec ByYmax) Swap(i, j int) { rvec[i], rvec[j] = rvec[j], rvec[i] }
func (rvec ByYmax) Less(i, j int) bool {
	h_nn_i := rvec[i].nnint.(*hart6_NN)
	h_nn_j := rvec[j].nnint.(*hart6_NN)
	return h_nn_i.ymax < h_nn_j.ymax
}

var rrvec []*NeuRunner

func (h_nn *hart6_NN) TrainStep(xvec []float64, yvec []float64) {
	h_nn.forward(xvec)
	if yvec[0] > h_nn.ymax {
		copyVector(h_nn.xmax, xvec)
		h_nn.ymax = yvec[0]
		h_nn.cost = h_nn.costfunction(yvec)
	}
	ynorm := h_nn.normalizeY(yvec)
	h_nn.computeDeltas(ynorm)
	h_nn.backpropDeltas()
	h_nn.backpropGradients()
}

func Test_hart6_1(t *testing.T) {
	input := NeuLayerConfig{size: 6}
	hidden := NeuLayerConfig{"tanh", input.size * 5}
	output := NeuLayerConfig{"sigmoid", 1}

	xvec := newVector(input.size)

	p := NewNeuParallel(1000)
	pnum := 6
	rrvec := make([]*NeuRunner, pnum)
	ttp := &TTP{maxbackprops: 1E6}

	for i := 1; i <= pnum; i++ {
		// NN
		tu := &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: XavierNewRand}
		nn := NewNeuNetwork(input, hidden, 3, output, tu)
		nn.layers[1].config.actfname = "relu"
		// hart6_NN
		h_nn := &hart6_NN{NeuNetworkInterface: nn, xmax: newVector(input.size), xsgn: newVector(input.size), xprv: newVector(input.size)}

		r := p.attach(h_nn, i, true, ttp)
		h_nn.initXavier(r.newrand)

		rrvec[i-1] = r
	}

	p.start() // go
	ppnum := 0
	eps := 0.001
	for {
		// 1. swap prev if better
		for k := 1; k <= pnum; k++ {
			r := p.get(k)
			h_nn := r.nnint.(*hart6_NN)
			yp := fn_hartmann(h_nn.xprv)
			h_nn.yprv = yp[0]
			y := fn_hartmann(h_nn.xmax)
			h_nn.ymax = y[0]
			if yp[0] > y[0] {
				h_nn.ymax, h_nn.yprv = h_nn.yprv, h_nn.ymax
				copyVector(xvec, h_nn.xmax)
				copyVector(h_nn.xmax, h_nn.xprv)
				copyVector(h_nn.xprv, xvec)
			}
		}

		// 2. sort
		sort.Sort(ByYmax(rrvec))
		// 3. sign
		for k := 1; k <= pnum; k++ {
			r := p.get(k)
			h_nn := r.nnint.(*hart6_NN)
			for j := 0; j < input.size; j++ {
				copyVector(xvec, h_nn.xmax)
				h_nn.xsgn[j] = 0
				y0 := fn_hartmann(xvec)
				xvec[j] += eps
				y1 := fn_hartmann(xvec)
				xvec[j] -= eps * 2
				y2 := fn_hartmann(xvec)
				if y1[0] > y0[0]+eps/10 && y1[0] > y2[0] {
					h_nn.xsgn[j] = 1
				} else if y2[0] > y0[0]+eps/10 && y2[0] > y1[0] {
					h_nn.xsgn[j] = -1
				}
			}
		}
		// 4. fill-in
		for k := 1; k <= pnum; k++ {
			r := p.get(k)
			i := 0
			// 3.1. 30% - random (0, 1)
			for ; i < p.size/3; i++ {
				fillVector(xvec, 0.0, 1.0)
				yvec := fn_hartmann(xvec)
				r.addSample(xvec, yvec)
			}
			// 3.2. based on sorted confidence
			for kk := ppnum - 1; kk >= 0; kk-- {
				rr := rrvec[kk]
				h_nn := rr.nnint.(*hart6_NN)
				fillsz := p.size / 35 * (kk + 1)
				std := 0.1
				for ii := 0; ii < fillsz && i < p.size; ii++ {
					fillVectorSpecial(xvec, h_nn.xmax, h_nn.xsgn, std, r.newrand)
					yvec := fn_hartmann(xvec)
					r.addSample(xvec, yvec)
					i++
				}
			}
			// 3.3. remaining if any - random (0, 1)
			for ; i < p.size; i++ {
				fillVector(xvec, 0.0, 1.0)
				yvec := fn_hartmann(xvec)
				r.addSample(xvec, yvec)
			}
		}
		// 5. store prev
		for k := 1; k <= pnum; k++ {
			r := p.get(k)
			h_nn := r.nnint.(*hart6_NN)
			h_nn.yprv = h_nn.ymax
			copyVector(h_nn.xprv, h_nn.xmax)
		}

		// 5. compute
		p.compute()

		ppnum = pnum
		// 6. log
		if p.step%100 == 0 {
			for k := 1; k <= pnum; k++ {
				r := p.get(k)
				h_nn := r.nnint.(*hart6_NN)
				xvec[k-1] = math.Sqrt(normL2DistanceSquared(h_nn.xmax, hart6_opt))
			}
			fmt.Printf("distances: %.4f\n", xvec)
			for k := 1; k <= pnum; k++ {
				r := p.get(k)
				h_nn := r.nnint.(*hart6_NN)
				xvec[k-1] = fn_hartmann(h_nn.xmax)[0]
			}
			fmt.Printf("ymax     : %.4f\n", xvec)
		}
	}
}

func Test_hart6_2(t *testing.T) {
	input := NeuLayerConfig{size: 6}
	hidden := NeuLayerConfig{"tanh", input.size * 5}
	output := NeuLayerConfig{"sigmoid", 1}

	xvec := newVector(input.size)

	p := NewNeuParallel(1000)
	pnum := 6
	rrvec := make([]*NeuRunner, pnum)
	ttp := &TTP{maxbackprops: 1E6}

	for i := 1; i <= pnum; i++ {
		// NN
		tu := &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: XavierNewRand}
		nn := NewNeuNetwork(input, hidden, 3, output, tu)
		nn.layers[1].config.actfname = "relu"
		// hart6_NN
		h_nn := &hart6_NN{NeuNetworkInterface: nn, xmax: newVector(input.size), xsgn: newVector(input.size), xprv: newVector(input.size)}

		r := p.attach(h_nn, i, true, ttp)
		h_nn.initXavier(r.newrand)

		rrvec[i-1] = r
	}

	p.start() // go
	std := 0.1
	// first time
	for k := 1; k <= pnum; k++ {
		r := p.get(k)
		for i := 0; i < p.size; i++ {
			fillVector(xvec, 0.0, 1.0)
			yvec := fn_hartmann(xvec)
			r.addSample(xvec, yvec)
		}
	}
	p.compute()

	for {
		// 1. max
		var ymax = []float64{-1.0}
		for k := 1; k <= pnum; k++ {
			r := p.get(k)
			h_nn := r.nnint.(*hart6_NN)
			y := fn_hartmann(h_nn.xmax)
			if y[0] > ymax[0] {
				ymax[0] = y[0]
				copyVector(xvec, h_nn.xmax)
			}
		}
		for k := 1; k <= pnum; k++ {
			r := p.get(k)
			h_nn := r.nnint.(*hart6_NN)
			copyVector(h_nn.xmax, xvec)
		}

		// 4. fill-in
		for k := 1; k <= pnum; k++ {
			r := p.get(k)
			h_nn := r.nnint.(*hart6_NN)
			i := 0
			for ; i < p.size/3; i++ {
				if i == 0 {
					r.addSample(h_nn.xmax, ymax)
				} else {
					fillVectorSpecial(xvec, h_nn.xmax, h_nn.xsgn, std, r.newrand)
					yvec := fn_hartmann(xvec)
					r.addSample(xvec, yvec)
				}
			}
			for ; i < p.size; i++ {
				fillVector(xvec, 0.0, 1.0)
				yvec := fn_hartmann(xvec)
				r.addSample(xvec, yvec)
			}

		}

		// 5. compute
		p.compute()

		// 6. log
		if p.step%100 == 0 {
			for k := 1; k <= pnum; k++ {
				r := p.get(k)
				h_nn := r.nnint.(*hart6_NN)
				xvec[k-1] = math.Sqrt(normL2DistanceSquared(h_nn.xmax, hart6_opt))
			}
			fmt.Printf("distances: %.4f\n", xvec)
			for k := 1; k <= pnum; k++ {
				r := p.get(k)
				h_nn := r.nnint.(*hart6_NN)
				xvec[k-1] = fn_hartmann(h_nn.xmax)[0]
			}
			fmt.Printf("ymax     : %.4f\n", xvec)
		}
	}
}

//

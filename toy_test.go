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

func fn_hartmann(xvec []float64) []float64 {
	yvec := []float64{0}
	for i := 0; i < 4; i++ {
		sum := 0.0
		for j := 0; j < 6; j++ {
			val := xvec[j] - hart6_mP[i][j]
			sum -= hart6_mA[i][j] * val * val
		}
		yvec[0] -= hart6_mC[i] * math.Exp(sum)
	}
	yvec[0] /= -3.33 // NOTE: normalize into (0, 1)
	return yvec
}

//=====================================================================
//
// Parallel compute: max(fn_hartmann) by multiple NN runners (experimental)
//
//=====================================================================
type hart6_NN struct {
	NeuNetworkInterface
	// sticks
	xmax []float64
	ymax float64
}

func Test_hartmann_max(t *testing.T) {
	// config: NN layers
	input := NeuLayerConfig{size: 6}
	hidden := NeuLayerConfig{"tanh", input.size * 5}
	output := NeuLayerConfig{"identity", 1}

	// config: hyper-params
	pnum := 5 // num NN runners (goroutines)
	if cli.int1 != 0 {
		pnum = cli.int1
	}
	sparse := 10 //
	if cli.int2 != 0 {
		sparse = cli.int2
	}
	gradstep := 0.001 //
	sigma := 0.01
	gradsmin := 2     //
	gradsmax := 10    //
	period := 100     // logging period
	numevals := 0     // <= maxevals (the budget)
	trainevals := 300 //
	if cli.int3 != 0 {
		trainevals = cli.int3
	}
	maxevals := 2300 //
	if cli.int4 != 0 {
		maxevals = cli.int4
	}
	var pd, ed, cnt1, cnt2 int

	// local & tmp
	xvec := newVector(input.size)
	yvec := []float64{0}
	xmax := newVector(input.size)
	ymax := []float64{-1.0}
	xinc := newVector(input.size)

	// parallel
	psize := trainevals / pnum
	psize = psize / 10 * 10
	//fmt.Printf("pnum=%d, psize=%d, sparse=%d, evals=%d\n", pnum, psize, sparse, maxevals)
	str := fmt.Sprintf("%2d, ", pnum)
	p := NewNeuParallel(psize)

	// inner functions
	fn_hartmann_max := func(xvec []float64) float64 {
		y := fn_hartmann(xvec)[0]
		if y > ymax[0] {
			ymax[0] = y
			copyVector(xmax, xvec)
		}
		numevals++
		// log
		if numevals/period != pd {
			pd = numevals / period

			q := math.Sqrt(normL2DistanceSquared(xmax, hart6_opt))
			str += fmt.Sprintf("%.3f, ", q)
			cnt1, cnt2 = 0, 0
		}
		return y
	}
	fn_gradstep := func(xvec []float64, h_nn *hart6_NN, dx float64) bool {
		nn := h_nn.NeuNetworkInterface.(*NeuNetwork)
		nonzero := nn.realGradSign(xinc, 0, xvec)
		if nonzero {
			mulVectorNum(xinc, dx)
			addVectorElem(xvec, xinc)
			for j := 0; j < h_nn.getIsize(); j++ {
				if xvec[j] < 0 || xvec[j] > 1 {
					return false
				}
			}
		}
		return nonzero
	}

	// construct NN runners
	ttp := &TTP{sequential: true, num: 10}
	for i := 1; i <= pnum; i++ {
		// NN
		tu := &NeuTunables{gdalgname: ADAM, batchsize: 10, winit: XavierNewRand}
		// diversify
		if i%3 == 1 {
			tu.gdalgname = RMSprop
		} else if i%3 == 2 {
			tu.gdalgname = Rprop
		}
		nn := NewNeuNetwork(input, hidden, 3, output, tu)
		nn.layers[1].config.actfname = "relu"

		// hart6_NN
		h_nn := &hart6_NN{NeuNetworkInterface: nn, xmax: newVector(input.size)}
		// runner
		r := p.attach(h_nn, i, true, ttp)
		nn.initXavier(r.newrand) // diversify
		nn.sparsify(r.newrand, sparse)
	}

	// go parallel
	p.start()
	defer p.stop()

	// pretrain #1: random
	for k := 1; k <= pnum; k++ {
		r := p.get(k)
		h_nn := r.nnint.(*hart6_NN)
		for i := 0; i < p.size; i++ {
			fillVector(xvec, 0.0, 1.0)
			yvec[0] = fn_hartmann_max(xvec)
			r.addSample(xvec, yvec)
			if yvec[0] > h_nn.ymax {
				h_nn.ymax = yvec[0]
				copyVector(h_nn.xmax, xvec)
			}
		}
	}
	p.compute()

	// pretrain #2: rotate
	for rotate := 1; rotate <= pnum-1; rotate++ {
		p.rotate()
		p.compute()
	}

	// main loop
	ed = numevals / trainevals
	for keepgoing := true; keepgoing && numevals <= maxevals; {
		for k := 1; k <= pnum && numevals <= maxevals; k++ {
			r := p.get(k)
			h_nn := r.nnint.(*hart6_NN)

			for kk := 1; kk <= pnum && numevals <= maxevals; kk++ {
				other_r := p.get(kk)
				other_nn := other_r.nnint.(*hart6_NN)

				copyVector(xvec, other_nn.xmax)
				for i := 0; i < gradsmax && numevals <= maxevals; i++ {
					if !fn_gradstep(xvec, h_nn, gradstep) {
						break
					}
					yvec[0] = fn_hartmann_max(xvec)
					r.addSample(xvec, yvec)
					if yvec[0] > other_nn.ymax {
						other_nn.ymax = yvec[0]
						copyVector(other_nn.xmax, xvec)
						cnt1++
					} else if i > gradsmin {
						break
					}
				}

				// gaussian(other xmax)
				for i := 0; i < 10 && numevals <= maxevals; i++ {
					y := other_nn.ymax
					fillVectorSpecial(xvec, other_nn.xmax, r.newrand, sigma)
					yvec[0] = fn_hartmann_max(xvec)
					if y < yvec[0] {
						copyVector(other_nn.xmax, xvec)
						other_nn.ymax = yvec[0]
						cnt2++
					}
					r.addSample(xvec, yvec)
				}
			}

			// reuse (rotate)
			if numevals/trainevals != ed {
				ed = numevals / trainevals
				for rotate := 1; rotate <= pnum-1; rotate++ {
					p.rotate()
					p.compute()
				}
			}
		}

		// 2. compute
		keepgoing = p.compute()
	}
	p.logger.Print(str[:len(str)-2])
}

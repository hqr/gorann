package gorann

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// test avg(previous, current)
func Test_pcavg(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 1}
	hidden := NeuLayerConfig{"sigmoid", 4}
	output := NeuLayerConfig{"sigmoid", 1}
	rnn := NewNaiveRnn(input, hidden, 2, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, winit: Xavier})
	rnn.layers[1].config.actfname = "tanh"
	ntrain, ngrad, ntest := 1000, rnn.tunables.batchsize, 8
	Xs, Ys := newMatrix(ntrain+ngrad+ntest, 1), newMatrix(ntrain+ngrad+ntest, 1)
	// simple historical dependency on the previous input
	ffill := func(i, iprev int) {
		Xs[i][0] = rand.Float64()
		Ys[i][0] = (Xs[iprev][0] + Xs[i][0]) / 2
	}

	ttp := &TTP{nn: rnn, resultset: Ys[:ntrain+ngrad], maxbackprops: 1E7, maxcost: 1E-4, sequential: true}
	var cnv int
	for cnv = 0; cnv == 0; {
		for i := 0; i < ntrain+ngrad; i++ {
			k := i - 1
			if i == 0 { // wrap around
				k = ntrain + ngrad - 1
			}
			ffill(i, k)
		}
		if cli.checkgrad && ttp.sequential && rnn.tunables.batchsize > 1 {
			cnv = ttp.Train(TtpArr(Xs[:ntrain]))
			l := rand.Int31n(int32(rnn.lastidx))
			layer := rnn.layers[l]
			next := layer.next
			i, j := rand.Int31n(int32(layer.size)), rand.Int31n(int32(next.size))
			ttp.trainAndCheckGradients(TtpArr(Xs), ntrain, ntrain+ngrad, int(l), int(i), int(j))
		} else {
			cnv = ttp.Train(TtpArr(Xs[:ntrain+ngrad]))
		}
	}
	mse := 0.0
	for i := 0; i < ntest; i++ {
		ffill(ntrain+ngrad+i, ntrain+ngrad+i-1)
	}
	for i := 0; i < ntest; i++ {
		j := ntrain + ngrad + i
		avec := rnn.Predict(Xs[j])
		mse += rnn.CostMse(Ys[j])
		fmt.Printf("(%.2f + %.2f)/2 -> %.2f : %.2f\n", Xs[j-1][0], Xs[j][0], avec[0], Ys[j][0])
	}
	fmt.Printf("mse %.5f (nbp %dK)\n", mse/4, rnn.getNbprops()/1000)
	if cnv&ConvergedMaxBackprops > 0 {
		t.Errorf("reached the maximum number of back propagations (%dK)\n", rnn.getNbprops()/1000)
	}
}

// exponential moving average (EMA)
func Test_emavg(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 1}
	hidden := NeuLayerConfig{"sigmoid", 4}
	output := NeuLayerConfig{"sigmoid", 1}
	// rnn := NewNaiveRnn(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, batchsize: 1, gdalgscopeall: true})
	rnn := NewLimitedRnn(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, batchsize: 1, gdalgscopeall: true, winit: Xavier}, 1)
	rnn.layers[1].config.actfname = "tanh"

	ntrain, ntest, alph := 1000, 8, 0.6
	Xs, Ys := newMatrix(ntrain+ntest, 1), newMatrix(ntrain+ntest, 1)
	// simple historical dependency on the previous output (state)
	ffill := func(i, iprev int) {
		Xs[i][0] = rand.Float64()
		Ys[i][0] = alph*Xs[i][0] + (1-alph)*Ys[iprev][0]
	}
	ttp := &TTP{nn: rnn, resultset: Ys[:ntrain], maxbackprops: 1E7, maxcost: 1E-5, sequential: true}
	var cnv int
	for cnv = 0; cnv == 0; {
		for i := 0; i < ntrain; i++ {
			k := i - 1
			if i == 0 { // wrap around
				k = ntrain - 1
			}
			ffill(i, k)
		}
		cnv = ttp.Train(TtpArr(Xs[:ntrain]))
	}
	mse := 0.0
	for i := 0; i < ntest; i++ {
		ffill(ntrain+i, ntrain+i-1)
	}
	for i := 0; i < ntest; i++ {
		j := ntrain + i
		avec := rnn.Predict(Xs[j])
		mse += rnn.CostMse(Ys[j])
		fmt.Printf("%.1f*%.2f + %.1f*%.2f -> %.2f : %.2f\n", alph, Xs[j][0], (1 - alph), Ys[j-1][0], avec[0], Ys[j][0])
	}
	fmt.Printf("mse %.5f (nbp %dK)\n", mse/4, rnn.getNbprops()/1000)
	if cnv&ConvergedCost == 0 {
		t.Errorf("failed to converge on cost (%d, %e)\n", cnv, ttp.maxcost)
	}
}

//
// new-state = F-polynomial(prev-input, prev-state, curr-input)
//
func Test_histpoly(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 2}
	hidden := NeuLayerConfig{"tanh", 8}
	output := NeuLayerConfig{"sigmoid", 2}
	// rnn := NewUnrolledRnn(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, batchsize: 10, gdalgscopeall: true})
	rnn := NewLimitedRnn(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, batchsize: 10, gdalgscopeall: true, winit: Xavier}, 2)

	ntrain, ntest, alph := 1000, 8, 0.6
	Xs, Ys := newMatrix(ntrain+ntest, 2), newMatrix(ntrain+ntest, 2)
	//
	// with non-linear dependency on both the previous vectored state and the previous input
	// (is it obfuscated enough?)
	//
	ffill := func(i, iprev int) {
		Xs[i][0], Xs[i][1] = rand.Float64(), rand.Float64()
		Ys[i][0] = alph*Xs[i][0] + (1-alph)*Ys[iprev][0]*(1-Xs[iprev][1])
		Ys[i][1] = alph*(1-Ys[iprev][1])*Xs[i][1] + (1-alph)*Xs[iprev][0]
	}
	ttp := &TTP{nn: rnn, resultset: Ys[:ntrain], maxbackprops: 1E7, maxcost: 1E-4, sequential: true}
	var cnv int
	for cnv = 0; cnv == 0; {
		for i := 0; i < ntrain; i++ {
			k := i - 1
			if i == 0 { // wrap around to maintain the sequence in order
				k = ntrain - 1
			}
			ffill(i, k)
		}
		cnv = ttp.Train(TtpArr(Xs[:ntrain]))
	}
	mse := 0.0
	for i := 0; i < ntest; i++ {
		ffill(ntrain+i, ntrain+i-1)
	}
	for i := 0; i < ntest; i++ {
		j := ntrain + i
		avec := rnn.Predict(Xs[j])
		mse += rnn.CostMse(Ys[j])
		fmt.Printf(" -> %3.2v : %3.2v\n", avec, Ys[j])
	}
	fmt.Printf("mse %.5f (nbp %dK)\n", mse/4, rnn.getNbprops()/1000)
	if cnv&ConvergedCost == 0 {
		t.Errorf("failed to converge on cost (%d, %e)\n", cnv, ttp.maxcost)
	}
}

//===================================================================================
//
// new-state = F-non-polynomial-fairly-obfuscated(prev-input, prev-state, curr-input)
//
//===================================================================================
func Test_sinhcosh(t *testing.T) {
	//
	// the /next/ stateful output is:
	// weighted sums of the current input and Sine/Cosine function of the product
	// of the previous input and the previous state...
	// and vise versa!
	//
	ffill := func(i, iprev int, Xs, Ys [][]float64) {
		Xs[i][0], Xs[i][1] = rand.Float64(), rand.Float64()
		Ys[i][0] = Xs[i][1] * math.Cosh(Ys[iprev][0]+Xs[iprev][0]) / 3
		Ys[i][1] = Xs[i][0] * math.Sinh(Ys[iprev][1]+Xs[iprev][1]) / 3
	}
	nonPoly_R2R2(t, ffill)
}

func Test_sinecosine(t *testing.T) {
	ffill := func(i, iprev int, Xs, Ys [][]float64) {
		Xs[i][0], Xs[i][1] = rand.Float64(), rand.Float64()
		Ys[i][0] = Xs[i][1] * math.Cos(Ys[iprev][0]+Xs[iprev][0])
		Ys[i][1] = Xs[i][0] * math.Sin(Ys[iprev][1]+Xs[iprev][1])
	}
	nonPoly_R2R2(t, ffill)
}

// https://en.wikipedia.org/wiki/Astroid
func Test_astroid(t *testing.T) {
	ffill := func(i, iprev int, Xs, Ys [][]float64) {
		Xs[i][0] = rand.Float64()
		q := math.Cos(Xs[i][0])
		Ys[i][0] = q * q * q
		q = math.Sin(Xs[i][0])
		Ys[i][1] = q * q * q
	}
	nonPoly_R2R2(t, ffill)
}

func nntestUnrolled() *UnrolledRnn {
	input := NeuLayerConfig{size: 2}
	output := NeuLayerConfig{"sigmoid", 2}
	hidden := NeuLayerConfig{"tanh", 8}
	return NewUnrolledRnn(input, hidden, 1, output, &NeuTunables{gdalgname: ADAM, batchsize: 100, gdalgscopeall: true, winit: Xavier})
}

func nntestNaive() *NaiveRnn {
	input := NeuLayerConfig{size: 2}
	output := NeuLayerConfig{"sigmoid", 2}
	hidden := NeuLayerConfig{"sigmoid", 4}
	return NewNaiveRnn(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, batchsize: 100, gdalgscopeall: true, winit: Xavier})
}

func nntestLimited(numconns int) *LimitedRnn {
	input := NeuLayerConfig{size: 2}
	output := NeuLayerConfig{"sigmoid", 2}
	hidden := NeuLayerConfig{"tanh", 8}
	return NewLimitedRnn(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, batchsize: 100, gdalgscopeall: true, winit: Xavier}, numconns)
}

func nonPoly_R2R2(t *testing.T, ffill func(i, iprev int, Xs, Ys [][]float64)) {
	rand.Seed(0)
	ntrain, ntest := 1000, 16
	Xs, Ys := newMatrix(ntrain+ntest, 2), newMatrix(ntrain+ntest, 2)
	var ttp *TTP
	if cli.lessrnn == 0 {
		fmt.Println("unrolled")
		rnn := nntestUnrolled()
		ttp = &TTP{nn: rnn, resultset: Ys[:ntrain], maxbackprops: 1E7, sequential: true}
	} else if cli.lessrnn >= 4 {
		fmt.Println("naive")
		rnn := nntestNaive()
		ttp = &TTP{nn: rnn, resultset: Ys[:ntrain], maxbackprops: 1E7, sequential: true}
	} else if cli.lessrnn < 0 {
		fmt.Println("mixed")
		rnn1 := nntestUnrolled()
		rnn2 := nntestNaive()
		rnn3 := nntestLimited(2) // 2 neuron conn-s
		mixer := NewWeightedGradientNN(rnn1, rnn2, rnn3)
		// mixer := NewWeightedMixerNN(rnn1, rnn2, rnn3)
		// mixer := NewWeightedMixerNN(rnn1, rnn2)
		ttp = &TTP{nn: mixer, resultset: Ys[:ntrain], maxbackprops: 1E7, sequential: true}
	} else {
		fmt.Println("limited", cli.lessrnn)
		rnn := nntestLimited(cli.lessrnn)
		ttp = &TTP{nn: rnn, resultset: Ys[:ntrain], maxbackprops: 1E7, sequential: true}
	}
	for cnv := 0; cnv == 0; {
		for i := 0; i < ntrain; i++ {
			k := i - 1
			if i == 0 { // wrap around to maintain the sequence in order
				k = ntrain - 1
			}
			ffill(i, k, Xs, Ys)
		}
		cnv = ttp.Train(TtpArr(Xs[:ntrain]))
	}
	mse := 0.0
	for i := 0; i < ntest; i++ {
		ffill(ntrain+i, ntrain+i-1, Xs, Ys)
	}
	for i := 0; i < ntest; i++ {
		j := ntrain + i
		avec := ttp.nn.Predict(Xs[j])
		mse += ttp.nn.costfunction(Ys[j])
		fmt.Printf(" -> %3.3v : %3.3v\n", avec, Ys[j])
	}
	fmt.Printf("mse %.5f (nbp %dK)\n", mse/4, ttp.nn.getNbprops()/1000)
}

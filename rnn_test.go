package gorann

import (
	"fmt"
	"math/rand"
	"testing"
)

// test avg(previous, current)
func Test_pcavg(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 1}
	hidden := NeuLayerConfig{"sigmoid", 4} // tanh
	output := NeuLayerConfig{"sigmoid", 1}
	rnn := NewNaiveRnn(input, hidden, 2, output, &NeuTunables{gdalgname: RMSprop, batchsize: 10, gdalgscopeall: true})
	rnn.initXavier()
	rnn.layers[1].config.actfname = "tanh"
	// rnn.tunables.momentum = 0
	ntrain, ngrad, ntest := 2000, rnn.tunables.batchsize, 8
	Xs, Ys := newMatrix(ntrain+ngrad+ntest, 1), newMatrix(ntrain+ngrad+ntest, 1)
	Xs[0][0], Ys[0][0] = rand.Float64(), Xs[0][0]/2
	for i := 1; i < len(Xs); i++ {
		Xs[i][0] = rand.Float64()
		Ys[i][0] = (Xs[i-1][0] + Xs[i][0]) / 2
	}
	converged := 0
	ttp := &TTP{nn: &rnn.NeuNetwork, resultset: Ys[:ntrain+ngrad], maxbackprops: 1E7, maxcost: 1E-4, sequential: true}
	for converged == 0 {
		if cli.checkgrad {
			converged = rnn.Train(Xs[:ntrain], ttp)
			l := rand.Int31n(int32(rnn.lastidx))
			layer := rnn.layers[l]
			next := layer.next
			i, j := rand.Int31n(int32(layer.size)), rand.Int31n(int32(next.size))
			rnn.Train_and_CheckGradients(Xs[ntrain:ntrain+ngrad], ttp, ntrain, int(l), int(i), int(j))
		} else {
			converged = rnn.Train(Xs[:ntrain+ngrad], ttp)
		}
	}
	mse := 0.0
	for i := 0; i < ntest; i++ {
		j := ntrain + i
		avec := rnn.Predict(Xs[j])
		mse += rnn.CostMse(Ys[j])
		fmt.Printf("(%.2f + %.2f)/2 -> %.2f : %.2f\n", Xs[j-1][0], Xs[j][0], avec[0], Ys[j][0])
	}
	fmt.Printf("mse %.5f (nbp %dK)\n", mse/4, rnn.nbackprops/1000)
	if converged&ConvergedMaxBackprops > 0 {
		t.Errorf("reached the maximum number of back propagations (%dK)\n", rnn.nbackprops/1000)
	}
}

// exponential moving average (EMA)
func Test_emavg(t *testing.T) {
	rand.Seed(0)
	input := NeuLayerConfig{size: 1}
	hidden := NeuLayerConfig{"sigmoid", 4} // tanh
	output := NeuLayerConfig{"sigmoid", 1}
	rnn := NewNaiveRnn(input, hidden, 2, output, &NeuTunables{gdalgname: ADAM, batchsize: 1, gdalgscopeall: true})
	rnn.layers[1].config.actfname = "tanh"
	rnn.initXavier()

	ntrain, ntest, alph := 8000, 8, 0.6
	Xs, Ys := newMatrix(ntrain+ntest, 1), newMatrix(ntrain+ntest, 1)
	Xs[0][0] = rand.Float64()
	for i := 1; i < len(Xs); i++ {
		Xs[i][0] = rand.Float64()
		Ys[i][0] = alph*Xs[i][0] + (1-alph)*Ys[i-1][0]
	}
	converged := 0
	ttp := &TTP{nn: &rnn.NeuNetwork, resultset: Ys[:ntrain], maxbackprops: 1E7, maxcost: 1E-4, sequential: true}
	for converged == 0 {
		converged = rnn.Train(Xs[:ntrain], ttp)
	}
	mse := 0.0
	for i := 0; i < ntest; i++ {
		j := ntrain + i
		avec := rnn.Predict(Xs[j])
		mse += rnn.CostMse(Ys[j])
		fmt.Printf("%.1f*%.2f + %.1f*%.2f -> %.2f : %.2f\n", alph, Xs[j][0], (1 - alph), Ys[j-1][0], avec[0], Ys[j][0])
	}
	fmt.Printf("mse %.5f (nbp %dK)\n", mse/4, rnn.nbackprops/1000)
	if converged&ConvergedCost == 0 {
		t.Errorf("failed to converge on cost (%d, %e)\n", converged, ttp.maxcost)
	}
}

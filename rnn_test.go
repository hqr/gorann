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
	rnn := NewNaiveRnn(input, hidden, 2, output, NeuTunables{gdalgname: RMSprop, batchsize: 10})
	rnn.initXavier()
	// rnn.tunables.momentum = 0
	ntrain, ntest := 1000, 8
	Xs, Ys := newMatrix(ntrain+ntest, 1), newMatrix(ntrain+ntest, 1)
	Xs[0][0], Ys[0][0] = rand.Float64(), Xs[0][0]/2
	for i := 1; i < len(Xs); i++ {
		Xs[i][0] = rand.Float64()
		Ys[i][0] = (Xs[i-1][0] + Xs[i][0]) / 2
	}
	converged := 0
	ttp := &TTP{nn: &rnn.NeuNetwork, resultset: Ys[:ntrain], maxgradnorm: 0.06, maxbackprops: 5E6, seqtail: true}
	for converged == 0 {
		converged = rnn.Train(Xs[:ntrain], ttp)
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

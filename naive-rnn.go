package gorann

import (
//	"log"
)

type NaiveRnn struct {
	NeuNetwork
	ht1         *NeuLayer
	nn_unrolled *NeuNetwork
	nffs        int
}

func NewNaiveRnn(cinput NeuLayerConfig, chidden NeuLayerConfig, numhidden int, coutput NeuLayerConfig, tunables NeuTunables) (rnn *NaiveRnn) {
	assert(numhidden > 0)
	nn := NewNeuNetwork(cinput, chidden, numhidden, coutput, tunables)
	inputL := nn.layers[0]
	ht1 := &NeuLayer{config: chidden, idx: inputL.idx, size: chidden.size, prev: inputL.prev, next: inputL.next}
	ht1.init(nn)

	// to reference nn.layers[] + ht1 - for fixWeights() convenience
	nn_unrolled := &NeuNetwork{cinput: cinput, chidden: chidden, coutput: coutput, tunables: nn.tunables}
	nn_unrolled.lastidx = numhidden + 2
	nn_unrolled.layers = make([]*NeuLayer, numhidden+3)
	nn_unrolled.layers[0] = nn.layers[0]
	nn_unrolled.layers[1] = ht1
	for l := 1; l <= nn.lastidx; l++ {
		nn_unrolled.layers[l+1] = nn.layers[l]
	}

	rnn = &NaiveRnn{*nn, ht1, nn_unrolled, 0}
	rnn.nnint = rnn
	return
}

func (rnn *NaiveRnn) forward(xvec []float64) []float64 {
	rnn.nffs++
	if rnn.nffs == 1 {
		return rnn.NeuNetwork.forward(xvec)
	}
	var xnorm []float64 = xvec
	if rnn.callbacks.normcbX != nil {
		xnorm = cloneVector(xvec)
		rnn.callbacks.normcbX(xnorm)
	}
	inputL := rnn.layers[0]
	copy(inputL.avec, xnorm)
	h1 := rnn.ht1.next
	copy(rnn.ht1.avec, h1.avec)

	rnn.reForward()
	outputL := rnn.layers[rnn.lastidx]
	return outputL.avec
}

func (rnn *NaiveRnn) reForward() {
	h1 := rnn.ht1.next
	inputL := rnn.layers[0]
	actF := activations[h1.config.actfname]
	for i := 0; i < h1.config.size; i++ {
		sumNewInput := mulColVec(inputL.weights, i, inputL.avec, inputL.size)
		sumHistory := mulColVec(rnn.ht1.weights, i, rnn.ht1.avec, rnn.ht1.size)
		h1.zvec[i] = sumNewInput + sumHistory
		h1.avec[i] = actF.f(h1.zvec[i]) // NOTE: recursive activation on the history part..
	}

	for l := 2; l <= rnn.lastidx; l++ {
		rnn.forwardLayer(rnn.layers[l])
	}
}

func (rnn *NaiveRnn) backprop(yvec []float64) {
	rnn.NeuNetwork.backprop(yvec)

	h1 := rnn.ht1.next
	assert(h1 == rnn.layers[1])
	for i := 0; i < rnn.ht1.size; i++ {
		for j := 0; j < h1.size; j++ {
			rnn.ht1.gradient[i][j] += rnn.ht1.avec[i] * h1.deltas[j]
		}
	}
}

func (rnn *NaiveRnn) fixGradients(batchsize int) {
	rnn.nn_unrolled.fixGradients(batchsize)
}

func (rnn *NaiveRnn) fixWeights(batchsize int) {
	rnn.nn_unrolled.fixWeights(batchsize)
}

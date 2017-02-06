package gorann

import (
//	"log"
)

//===========================================================================
//
// network with a single "recurrent" layer that remembers
// post-activated state of the very first hidden layer
//
//===========================================================================
type NaiveRnn struct {
	NeuNetwork
	ht1 *NeuLayer // h1 at time (t - 1)
}

func NewNaiveRnn(cinput NeuLayerConfig, chidden NeuLayerConfig, numhidden int, coutput NeuLayerConfig, tunables *NeuTunables) (rnn *NaiveRnn) {
	assert(numhidden > 0)
	nn := NewNeuNetwork(cinput, chidden, numhidden, coutput, tunables)
	inputL := nn.layers[0]
	ht1 := &NeuLayer{config: chidden, idx: inputL.idx, size: chidden.size, prev: inputL.prev, next: inputL.next}
	ht1.init(nn)

	rnn = &NaiveRnn{NeuNetwork: *nn, ht1: ht1}
	rnn.nnint = rnn
	return
}

func (rnn *NaiveRnn) forward(xvec []float64) []float64 {
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
		h1.avec[i] = actF.f(h1.zvec[i]) // recursive activation on the history part as well
	}
	assert(h1.idx == 1) // forwarded and activated above
	for l := 2; l <= rnn.lastidx; l++ {
		rnn.forwardLayer(rnn.layers[l])
	}
}

func (rnn *NaiveRnn) backpropDeltas(yvec []float64) {
	rnn.NeuNetwork.backpropDeltas(yvec)
}

func (rnn *NaiveRnn) backpropGradients() {
	rnn.NeuNetwork.backpropGradients()

	h1 := rnn.ht1.next
	assert(h1 == rnn.layers[1])
	for i := 0; i < rnn.ht1.size; i++ {
		for j := 0; j < h1.size; j++ {
			rnn.ht1.gradient[i][j] += rnn.ht1.avec[i] * h1.deltas[j]
		}
	}
}

func (rnn *NaiveRnn) fixGradients(batchsize int) {
	rnn.layers[0].next = rnn.ht1
	rnn.NeuNetwork.fixGradients(batchsize)
	rnn.layers[0].next = rnn.ht1.next
}

func (rnn *NaiveRnn) fixWeights(batchsize int) {
	rnn.layers[0].next = rnn.ht1
	rnn.NeuNetwork.fixWeights(batchsize)
	rnn.layers[0].next = rnn.ht1.next
}

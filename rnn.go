package gorann

import (
	"fmt"
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
	ilayer := nn.layers[0]
	ht1 := &NeuLayer{config: chidden, idx: ilayer.idx, size: chidden.size, prev: ilayer.prev, next: ilayer.next}
	ht1.init(nn)

	rnn = &NaiveRnn{NeuNetwork: *nn, ht1: ht1}
	rnn.nnint = rnn
	return
}

func (rnn *NaiveRnn) forward(xvec []float64) []float64 {
	var xnorm = xvec
	cb := rnn.getCallbacks()
	if cb != nil && cb.normcbX != nil {
		xnorm = cloneVector(xvec)
		cb.normcbX(xnorm)
	}
	ilayer := rnn.layers[0]
	copy(ilayer.avec, xnorm)
	h1 := rnn.ht1.next
	copy(rnn.ht1.avec, h1.avec)

	return rnn.nnint.reForward()
}

func (rnn *NaiveRnn) reForward() []float64 {
	h1 := rnn.ht1.next
	ilayer := rnn.layers[0]
	actF := activations[h1.config.actfname]
	for i := 0; i < h1.config.size; i++ {
		sumNewInput := mulColVec(ilayer.weights, i, ilayer.avec, ilayer.size)
		sumHistory := mulColVec(rnn.ht1.weights, i, rnn.ht1.avec, rnn.ht1.size)
		h1.zvec[i] = sumNewInput + sumHistory
		h1.avec[i] = actF.f(h1.zvec[i]) // recursive activation on the history part as well
	}
	assert(h1.idx == 1) // forwarded and activated above
	for l := 2; l <= rnn.lastidx; l++ {
		rnn.forwardLayer(rnn.layers[l])
	}
	olayer := rnn.layers[rnn.lastidx]
	return olayer.avec
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

//===========================================================================
//
// fully connected network with "recurrent" layers backing up (historically)
// each of the respective hidden layers
//
//===========================================================================
type UnrolledRnn struct {
	NeuNetwork
	rhlayers []*NeuLayer
}

func NewUnrolledRnn(cinput NeuLayerConfig, chidden NeuLayerConfig, numhidden int, coutput NeuLayerConfig, tunables *NeuTunables) (rnn *UnrolledRnn) {
	assert(numhidden > 0)
	nn := NewNeuNetwork(cinput, chidden, numhidden, coutput, tunables)

	rhlayers := make([]*NeuLayer, numhidden)
	for l := 0; l < numhidden; l++ {
		prevL := nn.layers[l]
		ht1 := &NeuLayer{config: chidden, idx: l, size: chidden.size, prev: prevL.prev, next: prevL.next}
		ht1.init(nn)
		rhlayers[l] = ht1
	}

	rnn = &UnrolledRnn{NeuNetwork: *nn, rhlayers: rhlayers}
	rnn.nnint = rnn
	return
}

func (rnn *UnrolledRnn) forward(xvec []float64) []float64 {
	var xnorm = xvec
	cb := rnn.getCallbacks()
	if cb != nil && cb.normcbX != nil {
		xnorm = cloneVector(xvec)
		cb.normcbX(xnorm)
	}
	ilayer := rnn.layers[0]
	numhidden := len(rnn.rhlayers)
	copy(ilayer.avec, xnorm)
	for l := 0; l < numhidden; l++ {
		h := rnn.layers[l+1]
		ht1 := rnn.rhlayers[l]
		copy(ht1.avec, h.avec)
	}

	return rnn.nnint.reForward()
}

func (rnn *UnrolledRnn) reForward() []float64 {
	numhidden := len(rnn.rhlayers)
	for l := 0; l < numhidden; l++ {
		currL := rnn.layers[l+1]
		actF := activations[currL.config.actfname]
		ht1 := rnn.rhlayers[l]
		prevL := rnn.layers[l]

		for i := 0; i < currL.config.size; i++ {
			sumNewInput := mulColVec(prevL.weights, i, prevL.avec, prevL.size)
			sumHistory := mulColVec(ht1.weights, i, ht1.avec, ht1.size)
			currL.zvec[i] = sumNewInput + sumHistory
			currL.avec[i] = actF.f(currL.zvec[i])
		}
	}
	for l := numhidden + 1; l <= rnn.lastidx; l++ {
		rnn.forwardLayer(rnn.layers[l])
	}
	olayer := rnn.layers[rnn.lastidx]
	return olayer.avec
}

func (rnn *UnrolledRnn) backpropGradients() {
	rnn.NeuNetwork.backpropGradients()

	numhidden := len(rnn.rhlayers)
	for l := 0; l < numhidden; l++ {
		h := rnn.layers[l+1]
		ht1 := rnn.rhlayers[l]
		for i := 0; i < ht1.size; i++ {
			for j := 0; j < h.size; j++ {
				ht1.gradient[i][j] += ht1.avec[i] * h.deltas[j]
			}
		}
	}
}

func (rnn *UnrolledRnn) fixGradients(batchsize int) {
	rnn.unroll()
	rnn.NeuNetwork.fixGradients(batchsize)
	rnn.rollb()
}

func (rnn *UnrolledRnn) fixWeights(batchsize int) {
	rnn.unroll()
	rnn.NeuNetwork.fixWeights(batchsize)
	rnn.rollb()
}

func (rnn *UnrolledRnn) unroll() {
	numhidden := len(rnn.rhlayers)
	for l := 0; l < numhidden; l++ {
		prevL := rnn.layers[l]
		ht1 := rnn.rhlayers[l]
		prevL.next = ht1
	}
}

func (rnn *UnrolledRnn) rollb() {
	numhidden := len(rnn.rhlayers)
	for l := 0; l < numhidden; l++ {
		prevL := rnn.layers[l]
		ht1 := rnn.rhlayers[l]
		prevL.next = ht1.next
	}
}

//===========================================================================
//
// modification of the UnrolledRnn (above)
// where some of the synapses are blocked and dead forever,
// and where only the specified number of historic synapses are carrying the weight,
// so to speak
//
//===========================================================================
type LimitedRnn struct {
	UnrolledRnn
	nalive int // num alive synapses
}

func NewLimitedRnn(cinput NeuLayerConfig, chidden NeuLayerConfig, numhidden int, coutput NeuLayerConfig, tunables *NeuTunables,
	nalive int) (limrnn *LimitedRnn) {
	rnn := NewUnrolledRnn(cinput, chidden, numhidden, coutput, tunables)
	assert(nalive < chidden.size && nalive > 0, fmt.Sprintf("invalid nalive param: (%d, %d)", nalive, chidden.size))

	limrnn = &LimitedRnn{UnrolledRnn: *rnn, nalive: nalive}
	limrnn.nnint = limrnn
	return
}

func (rnn *LimitedRnn) reForward() []float64 {
	numhidden := len(rnn.rhlayers)
	for l := 0; l < numhidden; l++ {
		currL := rnn.layers[l+1]
		actF := activations[currL.config.actfname]
		ht1 := rnn.rhlayers[l]
		prevL := rnn.layers[l]

		for i := 0; i < currL.config.size; i++ {
			sumNewInput := mulColVec(prevL.weights, i, prevL.avec, prevL.size)

			sumHistory := 0.0
			if rnn.nalive == 1 {
				// this simulates the feed thru
				// a single synapse connecting i-th neuron with its own counterpart at (t-1)
				sumHistory = ht1.weights[i][i] * ht1.avec[i]
			} else {
				cnt := 0
				for j := i; j < ht1.size && cnt < rnn.nalive; j++ {
					sumHistory += ht1.weights[j][i] * ht1.avec[j]
					cnt++
				}
				if cnt < rnn.nalive {
					for j := i - 1; j >= 0 && cnt < rnn.nalive; j-- {
						sumHistory += ht1.weights[j][i] * ht1.avec[j]
						cnt++
					}
				}
				assert(cnt == rnn.nalive)
			}

			currL.zvec[i] = sumNewInput + sumHistory
			currL.avec[i] = actF.f(currL.zvec[i])
		}
	}
	for l := numhidden + 1; l <= rnn.lastidx; l++ {
		rnn.forwardLayer(rnn.layers[l])
	}
	olayer := rnn.layers[rnn.lastidx]
	return olayer.avec
}

func (rnn *LimitedRnn) backpropGradients() {
	rnn.NeuNetwork.backpropGradients()

	numhidden := len(rnn.rhlayers)
	for l := 0; l < numhidden; l++ {
		h := rnn.layers[l+1]
		ht1 := rnn.rhlayers[l]
		for i := 0; i < ht1.size; i++ {
			if rnn.nalive == 1 {
				ht1.gradient[i][i] += ht1.avec[i] * h.deltas[i]
			} else {
				cnt := 0
				for j := i; j < h.size && cnt < rnn.nalive; j++ {
					ht1.gradient[i][j] += ht1.avec[i] * h.deltas[j]
					cnt++
				}
				if cnt < rnn.nalive {
					for j := i - 1; j >= 0 && cnt < rnn.nalive; j-- {
						ht1.gradient[i][j] += ht1.avec[i] * h.deltas[j]
						cnt++
					}
				}
				assert(cnt == rnn.nalive)
			}
		}
	}
}

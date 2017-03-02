package gorann

import (
	"fmt"
)

type WeightedMixerNN struct {
	NeuNetwork
	nns []NeuNetworkInterface
	// get optimized at rt
	weights []float64
	// tmp space
	tmpdeltas []float64
	// for convenience
	ulayer *NeuLayer
	olayer *NeuLayer
}

func NewWeightedMixerNN(nns ...NeuNetworkInterface) *WeightedMixerNN {
	assert(len(nns) > 1)
	osize := nns[0].getCoutput().size
	for i := 1; i < len(nns); i++ {
		assert(nns[0].getCinput().size == nns[i].getCinput().size)
		assert(osize == nns[i].getCoutput().size)
	}
	// NN base
	culayer := NeuLayerConfig{"identity", len(nns) * osize}
	coutput := NeuLayerConfig{"identity", osize}
	tunables := new(NeuTunables)
	copyStruct(tunables, nns[0].getTunables())
	nn := NewNeuNetwork(culayer, NeuLayerConfig{}, 0, coutput, tunables)

	// mixer
	mixer := &WeightedMixerNN{NeuNetwork: *nn, nns: nns}
	mixer.olayer = nn.layers[nn.lastidx]
	mixer.ulayer = nn.layers[0]
	mixer.ulayer.size = mixer.ulayer.config.size
	mixer.tmpdeltas = newVector(osize)
	// equal mixing weights
	mixer.weights = newVector(mixer.ulayer.size, 1/float64(len(nns)))

	mixer.nnint = mixer
	return mixer
}

// in the forward pass networks do the work
func (mixer *WeightedMixerNN) forward(xvec []float64) []float64 {
	assert(len(xvec) == mixer.cinput.size)
	var xnorm = xvec
	if mixer.callbacks.normcbX != nil {
		xnorm = cloneVector(xvec)
		mixer.callbacks.normcbX(xnorm)
	}
	for i := 0; i < len(mixer.nns); i++ {
		mixer.nns[i].populateInput(xnorm, false)
	}
	return mixer.reForward()
}

func (mixer *WeightedMixerNN) reForward() []float64 {
	osize := mixer.olayer.size
	fillVector(mixer.olayer.avec, 0)
	for i := 0; i < len(mixer.nns); i++ {
		avec := mixer.nns[i].reForward()
		ii := i * osize
		copy(mixer.ulayer.avec[ii:], avec)
		// ulayer => olayer pass (instead of forwardLayer())
		mulVectorElem(mixer.ulayer.avec[ii:ii+osize], mixer.weights[ii:])
		addVectorElem(mixer.olayer.avec, mixer.ulayer.avec[ii:])
	}
	return mixer.olayer.avec
}

func (mixer *WeightedMixerNN) computeDeltas(yvec []float64) []float64 {
	osize := mixer.olayer.size
	for i := 0; i < len(mixer.nns); i++ {
		deltas := mixer.nns[i].computeDeltas(yvec)
		copyVector(mixer.tmpdeltas, deltas)

		ii := i * osize
		mulVectorElem(mixer.tmpdeltas, mixer.weights[ii:])
		addVectorElem(mixer.ulayer.deltas[ii:ii+osize], mixer.tmpdeltas)
	}
	return nil
}

func (mixer *WeightedMixerNN) backpropDeltas() {
	mixer.nbackprops++
	for i := 0; i < len(mixer.nns); i++ {
		// alternatively, propagate already weighted deltas:
		//   deltas := mixer.ulayer.deltas[i*osize : (i+1)*osize]
		//   mixer.nns[i].populateDeltas(deltas)
		mixer.nns[i].backpropDeltas()
	}
}

func (mixer *WeightedMixerNN) backpropGradients() {
	for i := 0; i < len(mixer.nns); i++ {
		mixer.nns[i].backpropGradients()
	}
}

func (mixer *WeightedMixerNN) fixGradients(batchsize int) {
	for i := 0; i < len(mixer.nns); i++ {
		mixer.nns[i].fixGradients(batchsize)
	}
}

func (mixer *WeightedMixerNN) fixWeights(batchsize int) {
	osize := mixer.olayer.size
	//
	// compute delta contributions by the respective networks
	//
	fillVector(mixer.tmpdeltas, 0)
	for i := 0; i < len(mixer.nns); i++ {
		ii := i * osize
		addVectorElemAbs(mixer.tmpdeltas, mixer.ulayer.deltas[ii:])
	}
	// adjust mixing weights accordingly
	alpha := mixer.tunables.alpha
	for i := 0; i < len(mixer.nns); i++ {
		ii := i * osize
		divVectorElemAbs(mixer.ulayer.deltas[ii:ii+osize], mixer.tmpdeltas)
		for j := 0; j < osize; j++ {
			val := 1 - mixer.ulayer.deltas[ii+j]
			mixer.weights[ii+j] = (1-alpha)*mixer.weights[ii+j] + alpha*val
		}
	}
	fillVector(mixer.ulayer.deltas, 0)
	// FIXME: debug
	if mixer.nbackprops%cli.nbp == 0 {
		for i := 0; i < len(mixer.nns); i++ {
			ii := i * osize
			fmt.Printf("%.2v\n", mixer.weights[ii:ii+osize])
		}
	}

	for i := 0; i < len(mixer.nns); i++ {
		mixer.nns[i].fixWeights(batchsize)
	}
}

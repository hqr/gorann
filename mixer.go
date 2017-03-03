package gorann

import (
	"fmt"
)

type WeightedMixerNN struct {
	NeuNetwork
	nns []NeuNetworkInterface
	// get optimized at rt
	weights []float64
	// work space
	tmpdeltas  []float64
	tmpweights []float64
	tmpcost    []float64
	// for convenience
	ulayer *NeuLayer
	olayer *NeuLayer
	// config
	alpha         float64
	maxcostratio  float64
	maxdeltaratio float64
	epsilon       float64
	zeroweight    float64
	nbpival       int
	nbpupd        int
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
	// tmp
	mixer.tmpdeltas = newVector(osize)
	mixer.tmpweights = newVector(osize)
	mixer.tmpcost = newVector(len(nns))
	// equal mixing weights
	mixer.weights = newVector(mixer.ulayer.size, 1/float64(len(nns)))

	mixer.nnint = mixer

	// config (hardcoded)
	mixer.alpha = mixer.tunables.alpha
	mixer.maxcostratio = 2
	mixer.maxdeltaratio = 2
	mixer.nbpival = mixer.tunables.batchsize * 10
	if mixer.nbpival > 1000 {
		mixer.nbpival = mixer.tunables.batchsize
	}
	mixer.epsilon = 1E-8
	mixer.zeroweight = 0.1 / float64(len(mixer.nns))
	return mixer
}

// in the forward pass networks do the work
func (mixer *WeightedMixerNN) forward(xvec []float64) []float64 {
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
		mixer.tmpcost[i] += mixer.nns[i].costfunction(yvec)

		ii := i * osize
		// Variant #1
		deltas := mixer.nns[i].computeDeltas(yvec)
		copyVector(mixer.tmpdeltas, deltas)

		// Variant #2 - straight difference (avec - yvec) FIXME
		// mixer.nns[i].computeDeltas(yvec)
		// copyVector(mixer.tmpdeltas, mixer.ulayer.avec[ii:ii+osize])
		// subVectorElem(mixer.tmpdeltas, yvec)

		mulVectorElem(mixer.tmpdeltas, mixer.weights[ii:])
		addVectorElemAbs(mixer.ulayer.deltas[ii:ii+osize], mixer.tmpdeltas)
	}
	return nil
}

func (mixer *WeightedMixerNN) backpropDeltas() {
	mixer.nbackprops++
	for i := 0; i < len(mixer.nns); i++ {
		// FIXME: propagate weighted (portions of the) deltas
		// osize := mixer.olayer.size
		// mydeltas := mixer.ulayer.deltas[i*osize : (i+1)*osize]
		// mixer.nns[i].populateDeltas(mydeltas)
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
	if mixer.nbackprops-mixer.nbpupd >= mixer.nbpival {
		mixer.nbpupd = mixer.nbackprops
		mixer.fixMixingWeights()
	}

	for i := 0; i < len(mixer.nns); i++ {
		mixer.nns[i].fixWeights(batchsize)
	}
}

func (mixer *WeightedMixerNN) fixMixingWeights() {
	osize := mixer.olayer.size

	// 1. cost/min-cost ratios
	mincost := 10000.0
	for i := 0; i < len(mixer.nns); i++ {
		mincost = fmin(mincost, mixer.tmpcost[i])
	}
	divVectorNum(mixer.tmpcost, mincost+mixer.epsilon)
	// 2. delta "contributions" by the respective NNs
	copyVector(mixer.tmpdeltas, mixer.ulayer.deltas[:osize])
	for i := 1; i < len(mixer.nns); i++ {
		ii := i * osize
		addVectorElem(mixer.tmpdeltas, mixer.ulayer.deltas[ii:])
	}
	addVectorNum(mixer.tmpdeltas, mixer.epsilon)
	// 3. deltas and delta/min-delta ratios
	for i := 0; i < len(mixer.nns); i++ {
		ii := i * osize
		divVectorElemAbs(mixer.ulayer.deltas[ii:ii+osize], mixer.tmpdeltas)
	}
	for j := 0; j < osize; j++ {
		mindelta := 10000.0
		for i := 0; i < len(mixer.nns); i++ {
			ii := i * osize
			mindelta = fmin(mindelta, mixer.ulayer.deltas[ii+j])
		}
		for i := 0; i < len(mixer.nns); i++ {
			ii := i * osize
			mixer.ulayer.deltas[ii+j] /= mindelta
		}
	}
	// 4. adjust mixing weights - deltas
	for i := 0; i < len(mixer.nns); i++ {
		for j := 0; j < osize; j++ {
			ii := i * osize
			if mixer.ulayer.deltas[ii+j] < mixer.maxdeltaratio {
				val := 1 / mixer.ulayer.deltas[ii+j]
				mixer.weights[ii+j] += mixer.alpha * val
			}
		}
	}
	// 5. adjust mixing weights - cost
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.tmpcost[i] < mixer.maxcostratio {
			ii := i * osize
			for j := 0; j < osize; j++ {
				val := 1 / mixer.tmpcost[i]
				mixer.weights[ii+j] += mixer.alpha * val
			}
		}
	}

	// 6. finally, rebalance mixing weights
	copyVector(mixer.tmpweights, mixer.weights[:osize])
	for i := 1; i < len(mixer.nns); i++ {
		ii := i * osize
		addVectorElem(mixer.tmpweights, mixer.weights[ii:ii+osize])
	}
	repeat := false
	for i := 0; i < mixer.ulayer.size; i++ {
		j := i % osize
		mixer.weights[i] /= mixer.tmpweights[j]
		if mixer.weights[i] < mixer.zeroweight && mixer.weights[i] != 0 {
			repeat = true
			mixer.weights[i] = 0
		}
	}
	if repeat {
		copyVector(mixer.tmpweights, mixer.weights[:osize])
		for i := 1; i < len(mixer.nns); i++ {
			ii := i * osize
			addVectorElem(mixer.tmpweights, mixer.weights[ii:ii+osize])
		}
		for i := 0; i < mixer.ulayer.size; i++ {
			j := i % osize
			mixer.weights[i] /= mixer.tmpweights[j]
		}
	}

	fillVector(mixer.ulayer.deltas, 0)
	fillVector(mixer.tmpcost, 0)
	// FIXME: debug
	if mixer.nbackprops%cli.nbp == 0 {
		for i := 0; i < len(mixer.nns); i++ {
			ii := i * osize
			fmt.Printf("%.2v\n", mixer.weights[ii:ii+osize])
		}
	}
}

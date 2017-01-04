package gorann

import (
	"math"
)

type NeuNetwork struct {
	cinput    NeuLayerConfig
	chidden   NeuLayerConfig
	coutput   NeuLayerConfig
	tunables  NeuTunables
	callbacks NeuCallbacks
	//
	layers  []*NeuLayer
	lastidx int
}

type NeuLayer struct {
	config    NeuLayerConfig
	idx, size int
	prev      *NeuLayer
	next      *NeuLayer
	nn        *NeuNetwork
	// runtime
	avec        []float64
	zvec        []float64
	deltas      []float64
	weights     [][]float64
	gradient    [][]float64
	pregradient [][]float64 // previous value of the gradient, applied with momentum
	rmsgradient [][]float64 // average or moving average of the RMS(gradient)
	rmswupdates [][]float64 // moving average of the RMS(weight update) - Adadelta only
}

// c-tor
func NewNeuNetwork(cinput NeuLayerConfig, chidden NeuLayerConfig, numhidden int, coutput NeuLayerConfig, tunables NeuTunables) *NeuNetwork {
	nn := &NeuNetwork{cinput: cinput, chidden: chidden, coutput: coutput, tunables: tunables}

	nn.lastidx = numhidden + 1
	nn.layers = make([]*NeuLayer, numhidden+2)

	var prev *NeuLayer
	for idx := 0; idx <= nn.lastidx; idx++ {
		var layer *NeuLayer
		switch idx {
		case 0:
			layer = &NeuLayer{config: cinput, idx: idx, size: cinput.size + 1, prev: prev} // +bias
		case nn.lastidx:
			layer = &NeuLayer{config: coutput, idx: idx, size: coutput.size, prev: prev}
		default:
			layer = &NeuLayer{config: chidden, idx: idx, size: chidden.size + 1, prev: prev} // +bias
		}
		if prev != nil {
			prev.next = layer
		}
		nn.layers[idx] = layer
		prev = layer
	}
	for idx := 0; idx <= nn.lastidx; idx++ {
		layer := nn.layers[idx]
		layer.init(nn)
	}
	return nn
}

func (nn *NeuNetwork) copyWeights(from *NeuNetwork, reset bool) {
	assert(nn.lastidx == from.lastidx)
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		layer_from := from.layers[l]
		assert(layer.config.size == layer_from.config.size)

		copyMatrix(layer.weights, layer_from.weights)

		if reset {
			fillVector(layer.avec, 1.0)
			fillVector(layer.zvec, 1.0)
			fillVector(layer.deltas, 0.0)
			zeroMatrix(layer.gradient)
			zeroMatrix(layer.pregradient)
			zeroMatrix(layer.rmsgradient)
			zeroMatrix(layer.rmswupdates)
		}
	}
}

func (layer *NeuLayer) init(nn *NeuNetwork) {
	layer.avec = newVector(layer.size, 1.0)
	layer.zvec = newVector(layer.size, 1.0)
	layer.deltas = newVector(layer.size)
	layer.nn = nn
	if layer.next == nil {
		return
	}
	layer.weights = newMatrix(layer.size, layer.next.size, -1.0, 1.0)
	layer.gradient = newMatrix(layer.size, layer.next.size)
	layer.pregradient = newMatrix(layer.size, layer.next.size)
	layer.rmsgradient = newMatrix(layer.size, layer.next.size)
	// if nn.tunables.gdalgname == Adadelta {
	layer.rmswupdates = newMatrix(layer.size, layer.next.size)
	// }
}

func (layer *NeuLayer) isHidden() bool {
	return layer.idx > 0 && layer.next != nil
}

func (nn *NeuNetwork) forward(xvec []float64) []float64 {
	assert(len(xvec) == nn.cinput.size)
	var xnorm []float64 = xvec
	if nn.callbacks.normcbX != nil {
		xnorm = cloneVector(xvec)
		nn.callbacks.normcbX(xnorm)
	}
	inputL := nn.layers[0]
	copy(inputL.avec, xnorm) // copies up to the smaller num elems, bias remains intact
	for l := 1; l <= nn.lastidx; l++ {
		nn.forwardLayer(nn.layers[l])
	}
	outputL := nn.layers[nn.lastidx]
	return outputL.avec
}

func (nn *NeuNetwork) forwardLayer(layer *NeuLayer) {
	prev := layer.prev
	actF := activations[layer.config.actfname]
	for i := 0; i < layer.config.size; i++ {
		sum := mulColVec(prev.weights, i, prev.avec, prev.size)
		layer.zvec[i] = sum
		layer.avec[i] = actF.f(sum) // bias excepted
	}
}

func (nn *NeuNetwork) backprop(yvec []float64) {
	assert(len(yvec) == nn.coutput.size)
	//
	// compute deltas moving from the last layer back to the first
	//
	outputL := nn.layers[nn.lastidx]
	actF := activations[outputL.config.actfname]
	for i := 0; i < nn.coutput.size; i++ {
		if actF.dfy != nil {
			outputL.deltas[i] = actF.dfy(outputL.avec[i]) * (yvec[i] - outputL.avec[i])
		} else {
			outputL.deltas[i] = actF.dfx(outputL.zvec[i]) * (yvec[i] - outputL.avec[i])
		}
	}
	for l := nn.lastidx - 1; l > 0; l-- {
		layer := nn.layers[l]
		next := layer.next
		actF = activations[layer.config.actfname]
		for i := 0; i < layer.size; i++ {
			sum := mulRowVec(layer.weights, i, next.deltas, next.size)
			if actF.dfy != nil {
				layer.deltas[i] = sum * actF.dfy(layer.avec[i])
			} else {
				layer.deltas[i] = sum * actF.dfx(layer.zvec[i])
			}
		}
	}
	//
	// use deltas to recompute weight gradients
	//
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				layer.gradient[i][j] += layer.avec[i] * next.deltas[j]
			}
		}
	}
}

//
// apply changes resulted from back propagation and accumulated in gradient[][]
//
func (nn *NeuNetwork) fixWeights(batchsize int) {
	if batchsize != BatchSGD {
		for l := 0; l < nn.lastidx; l++ {
			layer := nn.layers[l]
			divElemMatrix(layer.gradient, float64(batchsize))
		}
	}
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				if l == nn.lastidx-1 || nn.tunables.gdalgscope&GDoptimizationScopeAll > 0 {
					layer.weightij_gdalg(i, j) // i-th neuron, j-th synapse
				} else {
					layer.weightij(i, j)
				}
			}
		}
	}
}

func (layer *NeuLayer) weightij(i int, j int) {
	nn := layer.nn
	alpha := nn.tunables.alpha
	layer.weights[i][j] = layer.weights[i][j] + alpha*layer.gradient[i][j]
	layer.weights[i][j] += nn.tunables.momentum * layer.pregradient[i][j]
	layer.pregradient[i][j] = layer.gradient[i][j]
	layer.gradient[i][j] = 0
}

func (layer *NeuLayer) weightij_gdalg(i int, j int) {
	nn := layer.nn
	alpha := nn.tunables.alpha
	gamma, g1, eps := Gamma, 1-Gamma, Epsilon // hyperparameters
	lambda := Lambda                          // regularization rate aka weight decay

	switch nn.tunables.gdalgname {
	case Adagrad:
		if layer.rmsgradient[i][j] == 0 {
			layer.rmsgradient[i][j] = math.Pow(layer.gradient[i][j], 2)
		} else {
			alpha /= (math.Sqrt(layer.rmsgradient[i][j] + eps))
			layer.rmsgradient[i][j] += math.Pow(layer.gradient[i][j], 2)
		}
	case Adadelta:
		if layer.rmsgradient[i][j] == 0 {
			layer.rmsgradient[i][j] = math.Pow(layer.gradient[i][j], 2)
			deltaij := alpha * layer.gradient[i][j]
			layer.rmswupdates[i][j] = math.Pow(deltaij, 2)
		} else {
			layer.rmsgradient[i][j] = gamma*layer.rmsgradient[i][j] + g1*math.Pow(layer.gradient[i][j], 2)
			alpha = math.Sqrt(layer.rmswupdates[i][j]+eps) / (math.Sqrt(layer.rmsgradient[i][j] + eps))
			deltaij := alpha * layer.gradient[i][j]
			layer.rmswupdates[i][j] = gamma*layer.rmswupdates[i][j] + g1*math.Pow(deltaij, 2)
		}
	case RMSprop:
		if layer.rmsgradient[i][j] == 0 {
			layer.rmsgradient[i][j] = math.Pow(layer.gradient[i][j], 2)
		} else {
			layer.rmsgradient[i][j] = gamma*layer.rmsgradient[i][j] + g1*math.Pow(layer.gradient[i][j], 2)
			alpha /= (math.Sqrt(layer.rmsgradient[i][j] + eps))
		}
	}
	// adjust the weight
	if layer.isHidden() && i < layer.config.size && lambda > 0 && nn.tunables.regularization > 0 {
		if nn.tunables.regularization&RegularizeL2 > 0 {
			layer.weights[i][j] = layer.weights[i][j]*(1-alpha*lambda) + alpha*layer.gradient[i][j]
		} else if nn.tunables.regularization&RegularizeL1 > 0 {
			if layer.weights[i][j] < 0 {
				layer.weights[i][j] += alpha * lambda
			} else if layer.weights[i][j] > 0 {
				layer.weights[i][j] -= alpha * lambda
			}
			layer.weights[i][j] += alpha * layer.gradient[i][j]
		}
	} else {
		layer.weights[i][j] = layer.weights[i][j] + alpha*layer.gradient[i][j]
		layer.weights[i][j] += nn.tunables.momentum * layer.pregradient[i][j]
	}
	layer.pregradient[i][j] = layer.gradient[i][j]
	layer.gradient[i][j] = 0
}

func (nn *NeuNetwork) CostLinear(yvec []float64) float64 {
	assert(len(yvec) == nn.coutput.size)
	var ynorm []float64 = yvec
	if nn.callbacks.normcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.normcbY(ynorm)
	}
	var err float64
	outputL := nn.layers[nn.lastidx]
	for i := 0; i < len(ynorm); i++ {
		delta := ynorm[i] - outputL.avec[i]
		err += delta * delta
	}
	return err / 2
}

func (nn *NeuNetwork) CostLogistic(yvec []float64) float64 {
	assert(len(yvec) == nn.coutput.size)
	var ynorm []float64 = yvec
	if nn.callbacks.normcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.normcbY(ynorm)
	}
	var err float64
	outputL := nn.layers[nn.lastidx]
	for i := 0; i < len(yvec); i++ {
		costi := ynorm[i]*math.Log(outputL.avec[i]) + (1-ynorm[i])*math.Log(1-outputL.avec[i])
		err += costi
	}
	return -err / 2
}

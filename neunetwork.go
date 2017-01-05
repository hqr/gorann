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
	layers     []*NeuLayer
	lastidx    int
	nbackprops int
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
	avegradient [][]float64 // average or moving average of the gradient - ADAM only
	// tracking
	weitrack []float64 // weight changes expressed as euclidean distances: sum(distance(neuron-curr, neuron-prev)))
	gratrack []float64 // past and current euclidean-norm(gradients)
	costrack []float64 // last so many cost (function) values
}

// c-tor
func NewNeuNetwork(cinput NeuLayerConfig, chidden NeuLayerConfig, numhidden int, coutput NeuLayerConfig, tunables NeuTunables) *NeuNetwork {
	nn := &NeuNetwork{cinput: cinput, chidden: chidden, coutput: coutput, tunables: tunables}

	nn.lastidx = numhidden + 1
	nn.layers = make([]*NeuLayer, numhidden+2)
	// construct layers
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
	if nn.tunables.batchsize < BatchSGD {
		nn.tunables.batchsize = BatchSGD
	}
	// algorithm-specific initialization
	nn.initgdalg(nn.tunables.gdalgname)

	// other settings via TBD CLI
	// nn.tunables.tracking = TrackWeightChanges | TrackGradientChanges
	// nn.tunables.gdalgscope = GDoptimizationScopeAll
	return nn
}

func (nn *NeuNetwork) copyNetwork(from *NeuNetwork, weightsonly bool) {
	assert(nn.lastidx == from.lastidx)
	if !weightsonly {
		copyStruct(nn.tunables, from.tunables)
		nn.initgdalg(nn.tunables.gdalgname)
	}
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		layer_from := from.layers[l]
		assert(layer.config.size == layer_from.config.size)

		copyMatrix(layer.weights, layer_from.weights)
		if weightsonly {
			continue
		}
		fillVector(layer.avec, 1.0)
		fillVector(layer.zvec, 1.0)
		fillVector(layer.deltas, 0.0)
		zeroMatrix(layer.gradient)
		zeroMatrix(layer.pregradient)
		zeroMatrix(layer.rmsgradient)

		layer.initgdalg(nn.tunables.gdalgname)
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

	layer.initgdalg(nn.tunables.gdalgname)
	layer.weitrack = newVector(10) // nn.tunables.tracking & TrackWeightChanges
	layer.gratrack = newVector(10) // nn.tunables.tracking & TrackGradientChanges
	layer.costrack = newVector(10) // nn.tunables.tracking & TrackCostChanges
}

func (nn *NeuNetwork) initgdalg(gdalgname string) {
	if gdalgname == ADAM {
		nn.tunables.gdalgalpha = ADAM_alpha
		nn.tunables.beta1 = ADAM_beta1
		nn.tunables.beta2 = ADAM_beta2
		nn.tunables.beta1_t = ADAM_beta1_t
		nn.tunables.beta2_t = ADAM_beta2_t
		nn.tunables.eps = ADAM_eps
	} else if gdalgname == Adagrad {
		nn.tunables.gdalgalpha = nn.tunables.alpha
		nn.tunables.eps = GDALG_eps
	} else if gdalgname == Adadelta || gdalgname == RMSprop {
		nn.tunables.gdalgalpha = nn.tunables.alpha
		nn.tunables.eps = GDALG_eps
		nn.tunables.gamma = GDALG_gamma
	}
}

func (layer *NeuLayer) initgdalg(gdalgname string) {
	if gdalgname == ADAM {
		layer.avegradient = newMatrix(layer.size, layer.next.size)
	}
	if gdalgname == Adadelta {
		layer.rmswupdates = newMatrix(layer.size, layer.next.size)
	}
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

// see for instance https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
func (nn *NeuNetwork) backprop(yvec []float64) {
	assert(len(yvec) == nn.coutput.size)
	nn.nbackprops++
	//
	// compute deltas moving from the last layer back to the first
	//
	outputL := nn.layers[nn.lastidx]
	actF := activations[outputL.config.actfname]
	for i := 0; i < nn.coutput.size; i++ {
		if actF.dfy != nil {
			outputL.deltas[i] = actF.dfy(outputL.avec[i]) * (outputL.avec[i] - yvec[i])
		} else {
			outputL.deltas[i] = actF.dfx(outputL.zvec[i]) * (outputL.avec[i] - yvec[i])
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
	var dotrackg, dotrackw bool
	if nn.nbackprops%100 == 0 {
		dotrackw = nn.tunables.tracking&TrackWeightChanges > 0
		dotrackg = nn.tunables.tracking&TrackGradientChanges > 0
	}
	if nn.tunables.gdalgname == ADAM {
		nn.tunables.beta1_t *= nn.tunables.beta1
		nn.tunables.beta2_t *= nn.tunables.beta2
	}
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		var prew [][]float64
		if dotrackw {
			prew = cloneMatrix(layer.weights)
		}
		if dotrackg {
			edist0 := normL2Matrix(layer.gradient, nil)
			shiftVector(layer.gratrack)
			pushVector(layer.gratrack, edist0)
		}
		// move the weights in the direction opposite to the corresponding gradient vectors
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				// l-th layer, i-th neuron, j-th synapse
				var weightij float64
				if l == nn.lastidx-1 || nn.tunables.gdalgscope&GDoptimizationScopeAll > 0 {
					switch nn.tunables.gdalgname {
					case Adagrad:
						weightij = layer.weightij_Adagrad(i, j)
					case Adadelta:
						weightij = layer.weightij_Adadelta(i, j)
					case RMSprop:
						weightij = layer.weightij_RMSprop(i, j)
					case ADAM:
						weightij = layer.weightij_ADAM(i, j)
					}
				} else {
					weightij = layer.weightij(i, j)
				}
				layer.weights[i][j] = weightij
				layer.pregradient[i][j] = layer.gradient[i][j]
				layer.gradient[i][j] = 0
			}
		}
		if dotrackw {
			edist := normL2Matrix(prew, layer.weights)
			shiftVector(layer.weitrack)
			pushVector(layer.weitrack, edist)
		}
	}
}

func (layer *NeuLayer) weightij(i int, j int) float64 {
	nn := layer.nn
	alpha := nn.tunables.alpha
	weightij := layer.weights[i][j]
	weightij = weightij - alpha*layer.gradient[i][j]
	weightij -= nn.tunables.momentum * layer.pregradient[i][j]
	return weightij
}

func (layer *NeuLayer) weightij_Adagrad(i int, j int) float64 {
	nn := layer.nn
	alpha := nn.tunables.gdalgalpha
	weightij := layer.weights[i][j]
	eps := nn.tunables.eps

	if layer.rmsgradient[i][j] == 0 {
		layer.rmsgradient[i][j] = math.Pow(layer.gradient[i][j], 2)
	} else {
		alpha /= (math.Sqrt(layer.rmsgradient[i][j] + eps))
		layer.rmsgradient[i][j] += math.Pow(layer.gradient[i][j], 2)
	}

	weightij = weightij - alpha*layer.gradient[i][j]
	weightij -= nn.tunables.momentum * layer.pregradient[i][j]
	return weightij
}

func (layer *NeuLayer) weightij_Adadelta(i int, j int) float64 {
	nn := layer.nn
	alpha := nn.tunables.gdalgalpha
	weightij := layer.weights[i][j]
	gamma, g1, eps := nn.tunables.gamma, 1-nn.tunables.gamma, nn.tunables.eps

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

	weightij = weightij - alpha*layer.gradient[i][j]
	weightij -= nn.tunables.momentum * layer.pregradient[i][j]
	return weightij
}

func (layer *NeuLayer) weightij_RMSprop(i int, j int) float64 {
	nn := layer.nn
	alpha := nn.tunables.gdalgalpha
	weightij := layer.weights[i][j]
	gamma, g1, eps := nn.tunables.gamma, 1-nn.tunables.gamma, nn.tunables.eps

	if layer.rmsgradient[i][j] == 0 {
		layer.rmsgradient[i][j] = math.Pow(layer.gradient[i][j], 2)
	} else {
		layer.rmsgradient[i][j] = gamma*layer.rmsgradient[i][j] + g1*math.Pow(layer.gradient[i][j], 2)
		alpha /= (math.Sqrt(layer.rmsgradient[i][j] + eps))
	}
	weightij = weightij - alpha*layer.gradient[i][j]
	weightij -= nn.tunables.momentum * layer.pregradient[i][j]
	return weightij
}

func (layer *NeuLayer) weightij_ADAM(i int, j int) float64 {
	nn := layer.nn
	alpha := nn.tunables.gdalgalpha
	weightij := layer.weights[i][j]

	if layer.rmsgradient[i][j] == 0 {
		layer.rmsgradient[i][j] = math.Pow(layer.gradient[i][j], 2)
		layer.avegradient[i][j] = layer.gradient[i][j]
	} else {
		layer.avegradient[i][j] = nn.tunables.beta1*layer.avegradient[i][j] + (1-nn.tunables.beta1)*layer.gradient[i][j]
		layer.rmsgradient[i][j] = nn.tunables.beta2*layer.rmsgradient[i][j] + (1-nn.tunables.beta2)*math.Pow(layer.gradient[i][j], 2)
		alpha *= math.Sqrt(1-nn.tunables.beta2_t) / (1 - nn.tunables.beta1_t)
		weightij = weightij - alpha*layer.avegradient[i][j]/(math.Sqrt(layer.rmsgradient[i][j])+nn.tunables.eps)
	}
	return weightij
}

//
// cost and loss helper functions; in all 3 the yvec is true value
// assumption/requirement therefore: called after the corresponding feed-forward step
//
func (nn *NeuNetwork) CostLinear(yvec []float64) float64 {
	assert(len(yvec) == nn.coutput.size)
	var ynorm []float64 = yvec
	if nn.callbacks.normcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.normcbY(ynorm)
	}
	outputL := nn.layers[nn.lastidx]
	return normL2VectorSquared(ynorm, outputL.avec) / 2
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
	for i := 0; i < len(ynorm); i++ {
		costi := ynorm[i]*math.Log(outputL.avec[i]) + (1-ynorm[i])*math.Log(1-outputL.avec[i])
		err += costi
	}
	return -err / 2
}

// unlike the conventional Cost functions above, this one works on denormalized result vectors
// note also that it returns L1 norm
func (nn *NeuNetwork) AbsError(yvec []float64) float64 {
	assert(len(yvec) == nn.coutput.size)
	outputL := nn.layers[nn.lastidx]
	var ydenorm []float64 = outputL.avec
	if nn.callbacks.denormcbY != nil {
		ydenorm = cloneVector(outputL.avec)
		nn.callbacks.denormcbY(ydenorm)
	}
	return normL1Vector(ydenorm, yvec)
}

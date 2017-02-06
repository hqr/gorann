package gorann

import (
	"math"
	"math/rand"
)

//
// interfaces
//
type NeuNetworkInterface interface {
	forward(xvec []float64) []float64
	reForward()
	backpropDeltas(yvec []float64)
	backpropGradients()
	fixGradients(batchsize int)
	fixWeights(batchsize int)
}

//
// objects
//
type NeuNetwork struct {
	cinput    NeuLayerConfig
	chidden   NeuLayerConfig
	coutput   NeuLayerConfig
	tunables  *NeuTunables
	callbacks NeuCallbacks
	//
	layers     []*NeuLayer
	lastidx    int
	nbackprops int
	// polymorphism
	nnint NeuNetworkInterface
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
	wupdates    [][]float64 // weight updates - Rprop only
}

// c-tor
func NewNeuNetwork(cinput NeuLayerConfig, chidden NeuLayerConfig, numhidden int, coutput NeuLayerConfig, tunables *NeuTunables) *NeuNetwork {
	nn := &NeuNetwork{cinput: cinput, chidden: chidden, coutput: coutput, tunables: tunables}
	nn.nnint = nn
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
	// set  defaults (unless already defined)
	nn.initunables()
	// algorithm-specific hyper-params
	nn.initgdalg()

	for idx := 0; idx <= nn.lastidx; idx++ {
		layer := nn.layers[idx]
		layer.init(nn)
	}
	//
	// other useful settings
	//
	// nn.tunables.gdalgscopeall = true
	// nn.tunables.costfname = CostCrossEntropy
	// nn.tunables.lambda = DEFAULT_lambda
	// nn.tunables.batchsize = 10..100
	return nn
}

//
// methods
//
func (nn *NeuNetwork) initunables() {
	// defaults
	if nn.tunables.alpha == 0 {
		nn.tunables.alpha = DEFAULT_alpha
	}
	if nn.tunables.momentum == 0 {
		nn.tunables.momentum = DEFAULT_momentum
	}
	if nn.tunables.batchsize == 0 {
		nn.tunables.batchsize = DEFAULT_batchsize
	}
	if len(nn.tunables.costfname) == 0 {
		nn.tunables.costfname = CostMse
	}
	// command-line: override defaults and hardcodings
	if cli.alpha > 0 {
		nn.tunables.alpha = cli.alpha
	}
	if cli.momentum > 0 {
		nn.tunables.momentum = cli.momentum
	}
	if cli.lambda > 0 {
		nn.tunables.lambda = cli.lambda
	}
	if cli.batchsize > 0 {
		nn.tunables.batchsize = cli.batchsize
	}
	if len(cli.gdalgname) > 0 {
		nn.tunables.gdalgname = cli.gdalgname
	}
}

func (nn *NeuNetwork) initgdalg() {
	gdalgname := nn.tunables.gdalgname
	if gdalgname == ADAM {
		nn.tunables.gdalgalpha = ADAM_alpha
		nn.tunables.beta1 = ADAM_beta1
		nn.tunables.beta2 = ADAM_beta2
		nn.tunables.beta1_t = ADAM_beta1_t
		nn.tunables.beta2_t = ADAM_beta2_t
		nn.tunables.eps = ADAM_eps
	} else if gdalgname == Rprop {
		nn.tunables.eta = Rprop_eta
		nn.tunables.neta = Rprop_neta
	} else if gdalgname == Adagrad {
		nn.tunables.eps = DEFAULT_eps
	} else if gdalgname == Adadelta || gdalgname == RMSprop {
		nn.tunables.eps = DEFAULT_eps
		nn.tunables.gamma = DEFAULT_gamma
	}
}

func (layer *NeuLayer) init(nn *NeuNetwork) {
	layer.avec = newVector(layer.size)
	layer.zvec = newVector(layer.size)
	layer.deltas = newVector(layer.size)
	layer.nn = nn
	if layer.next == nil {
		assert(layer.config.actfname != "softmax" || nn.tunables.costfname == CostCrossEntropy, "softmax must be used with cross-entropy cost")
		return
	}
	// bias
	if layer.size > layer.config.size {
		assert(layer.size == layer.config.size+1)
		layer.avec[layer.config.size] = 1
	}
	// nor is it recommended
	assert(layer.config.actfname != "softmax", "softmax activation for inner layers is currently not supported")
	layer.weights = newMatrix(layer.size, layer.next.size, -1.0, 1.0)
	next := layer.next
	// init weights, alt init below
	for i := 0; i < layer.size; i++ {
		for j := 0; j < next.size; j++ {
			if layer.weights[i][j] >= 0 && layer.weights[i][j] < 0.5 {
				layer.weights[i][j] = 1 - layer.weights[i][j]
			} else if layer.weights[i][j] < 0 && layer.weights[i][j] > -0.5 {
				layer.weights[i][j] = -(1 + layer.weights[i][j])
			}
		}
	}
	layer.gradient = newMatrix(layer.size, layer.next.size)
	layer.pregradient = newMatrix(layer.size, layer.next.size)
	layer.rmsgradient = newMatrix(layer.size, layer.next.size)
	//
	// auxiliary GD matrices (NOTE: gdalgname)
	gdalgname := nn.tunables.gdalgname
	if gdalgname == ADAM {
		layer.avegradient = newMatrix(layer.size, layer.next.size)
	} else if gdalgname == Adadelta {
		layer.rmswupdates = newMatrix(layer.size, layer.next.size)
	} else if gdalgname == Rprop {
		layer.wupdates = newMatrix(layer.size, layer.next.size)
	}
}

func (nn *NeuNetwork) reset() {
	nn.initgdalg()
	gdalgname := nn.tunables.gdalgname
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		fillVector(layer.avec, 0)
		fillVector(layer.zvec, 0)
		if layer.size > layer.config.size {
			layer.avec[layer.config.size] = 1
		}
		fillVector(layer.deltas, 0.0)
		zeroMatrix(layer.gradient)
		zeroMatrix(layer.pregradient)
		zeroMatrix(layer.rmsgradient)
		// auxiliary matrices
		if gdalgname == ADAM {
			layer.avegradient = newMatrix(layer.size, layer.next.size)
		} else if gdalgname == Adadelta {
			layer.rmswupdates = newMatrix(layer.size, layer.next.size)
		} else if gdalgname == Rprop {
			layer.wupdates = newMatrix(layer.size, layer.next.size)
		}
	}
}

func (nn *NeuNetwork) copyNetwork(from *NeuNetwork) {
	assert(nn.lastidx == from.lastidx)
	copyStruct(nn.tunables, from.tunables)
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		layer_from := from.layers[l]
		assert(layer.config.size == layer_from.config.size)

		copyMatrix(layer.weights, layer_from.weights)
	}
}

// Xavier et al at http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
func (nn *NeuNetwork) initXavier() {
	qsix := math.Sqrt(6)
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		u := qsix / float64(layer.size+next.size)
		d := u * 2
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				layer.weights[i][j] = d*rand.Float64() - u
			}
		}
	}
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

// feed-forward pass on the *already stored* input, possibly with different weights
func (nn *NeuNetwork) reForward() {
	for l := 1; l <= nn.lastidx; l++ {
		nn.forwardLayer(nn.layers[l])
	}
}

func (nn *NeuNetwork) forwardLayer(layer *NeuLayer) {
	prev := layer.prev
	actF := activations[layer.config.actfname]
	if layer.config.actfname == "softmax" {
		sumexp := 0.0
		for i := 0; i < layer.config.size; i++ {
			sum := mulColVec(prev.weights, i, prev.avec, prev.size)
			layer.zvec[i] = sum
			sumexp += math.Exp(sum)
		}
		for i := 0; i < layer.config.size; i++ {
			layer.avec[i] = softmax(sumexp, layer.zvec[i])
		}
		return
	}
	for i := 0; i < layer.config.size; i++ {
		sum := mulColVec(prev.weights, i, prev.avec, prev.size)
		layer.zvec[i] = sum
		layer.avec[i] = actF.f(sum) // bias excepted
	}
}

func (nn *NeuNetwork) backpropDeltas(yvec []float64) {
	assert(len(yvec) == nn.coutput.size)
	outputL := nn.layers[nn.lastidx]
	cname, aname := nn.tunables.costfname, outputL.config.actfname
	assert(cname == CostMse || cname == CostCrossEntropy, "NIY: delta rule for the spec-ed cost function")

	nn.nbackprops++
	//
	// delta rules
	//
	actF := activations[aname]
	if cname == CostCrossEntropy && (aname == "sigmoid" || aname == "softmax") {
		for i := 0; i < nn.coutput.size; i++ {
			outputL.deltas[i] = outputL.avec[i] - yvec[i]
		}
	} else {
		for i := 0; i < nn.coutput.size; i++ {
			if cname == CostCrossEntropy {
				if nn.coutput.size == 1 {
					outputL.deltas[i] = (outputL.avec[i] - yvec[i]) / (outputL.avec[i] * (1 - outputL.avec[i]))
				} else {
					outputL.deltas[i] = -yvec[i] / outputL.avec[i]
				}
			} else {
				outputL.deltas[i] = outputL.avec[i] - yvec[i]
			}
			if actF.dfy != nil {
				outputL.deltas[i] *= actF.dfy(outputL.avec[i])
			} else {
				outputL.deltas[i] *= actF.dfx(outputL.zvec[i])
			}
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
}

//
// use the deltas (above) to recompute weight gradients
//
func (nn *NeuNetwork) backpropGradients() {
	for layer := nn.layers[0]; layer.next != nil; layer = layer.next {
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				layer.gradient[i][j] += layer.avec[i] * next.deltas[j]
			}
		}
	}
}

//
// step 1 applying changes resulted from back propagation and accumulated in the gradient[][]s
//
func (nn *NeuNetwork) fixGradients(batchsize int) {
	if batchsize > 1 {
		for layer := nn.layers[0]; layer.next != nil; layer = layer.next {
			divElemMatrix(layer.gradient, float64(batchsize))
		}
	}
}

// step 2 --/--/--
func (nn *NeuNetwork) fixWeights(batchsize int) {
	if nn.tunables.gdalgname == ADAM {
		nn.tunables.beta1_t *= nn.tunables.beta1
		nn.tunables.beta2_t *= nn.tunables.beta2
	}
	for layer := nn.layers[0]; layer.next != nil; layer = layer.next {
		next := layer.next
		// move the weights in the direction opposite to the corresponding gradient vectors
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				// l-th layer, i-th neuron, j-th synapse
				var weightij float64
				if layer.idx == nn.lastidx-1 || nn.tunables.gdalgscopeall {
					switch nn.tunables.gdalgname {
					case Adagrad:
						weightij = layer.weightij_Adagrad(i, j)
					case Adadelta:
						weightij = layer.weightij_Adadelta(i, j)
					case RMSprop:
						weightij = layer.weightij_RMSprop(i, j)
					case ADAM:
						weightij = layer.weightij_ADAM(i, j)
					case Rprop:
						weightij = layer.weightij_Rprop(i, j)
					}
				} else {
					weightij = layer.weightij(i, j)
				}
				// regularization
				if nn.tunables.lambda > 0 && i < layer.config.size {
					weightij -= nn.tunables.lambda * layer.weights[i][j] / float64(batchsize)
				}
				layer.weights[i][j] = weightij
				layer.pregradient[i][j] = layer.gradient[i][j]
				layer.gradient[i][j] = 0
			}
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

// ADAptive GRADient
func (layer *NeuLayer) weightij_Adagrad(i int, j int) float64 {
	nn := layer.nn
	alpha := nn.tunables.alpha
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
	alpha := nn.tunables.alpha
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
	// NOTE: still using momentum
	weightij -= nn.tunables.momentum * layer.pregradient[i][j]
	return weightij
}

// Root Mean Squared adaptive backPROPagation
func (layer *NeuLayer) weightij_RMSprop(i int, j int) float64 {
	nn := layer.nn
	alpha := nn.tunables.alpha
	weightij := layer.weights[i][j]
	gamma, g1, eps := nn.tunables.gamma, 1-nn.tunables.gamma, nn.tunables.eps

	if layer.rmsgradient[i][j] == 0 {
		layer.rmsgradient[i][j] = math.Pow(layer.gradient[i][j], 2)
	} else {
		layer.rmsgradient[i][j] = gamma*layer.rmsgradient[i][j] + g1*math.Pow(layer.gradient[i][j], 2)
		alpha /= (math.Sqrt(layer.rmsgradient[i][j] + eps))
	}
	weightij = weightij - alpha*layer.gradient[i][j]
	// NOTE: still using momentum
	weightij -= nn.tunables.momentum * layer.pregradient[i][j]
	return weightij
}

// ADAptive Moment estimation
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
		div := math.Sqrt(layer.rmsgradient[i][j]) + nn.tunables.eps
		weightij = weightij - alpha*layer.avegradient[i][j]/div
	}
	return weightij
}

// Resilient backPROPagation
func (layer *NeuLayer) weightij_Rprop(i int, j int) float64 {
	nn := layer.nn
	alpha := nn.tunables.alpha
	weightij := layer.weights[i][j]

	if layer.wupdates[i][j] == 0 {
		layer.wupdates[i][j] = math.Abs(alpha * layer.gradient[i][j])
		weightij = weightij - alpha*layer.gradient[i][j]
		return weightij
	}
	if layer.gradient[i][j] > 0 && layer.pregradient[i][j] > 0 {
		layer.wupdates[i][j] *= nn.tunables.eta
	} else if layer.gradient[i][j] < 0 && layer.pregradient[i][j] < 0 {
		layer.wupdates[i][j] *= nn.tunables.eta
	} else {
		layer.wupdates[i][j] *= nn.tunables.neta
	}
	if layer.gradient[i][j] > 0 {
		weightij = weightij - layer.wupdates[i][j]
	} else {
		weightij = weightij + layer.wupdates[i][j]
	}

	return weightij
}

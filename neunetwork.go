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
	reForward() []float64
	computeDeltas(yvec []float64) []float64
	backpropDeltas()
	backpropGradients()
	fixGradients(batchsize int)
	fixWeights(batchsize int)
	// global redirect, convenience
	Predict(xvec []float64) []float64
	TrainStep(xvec []float64, yvec []float64)
	// cost
	costfunction(yvec []float64) (cost float64)
	costl2reg() (creg float64)
	costl2_weightij(l int, i int, j int, eps float64) float64
	// aux
	normalizeY(yvec []float64) []float64
	populateInput(xvec []float64, normalize bool)
	populateDeltas(deltas []float64)
	initXavier(newrand *rand.Rand)
	// Get accessors
	getLayer(l int) *NeuLayer
	getIsize() int
	getOsize() int
	getHsize() int
	getHnum() int
	getTunables() *NeuTunables
	getCallbacks() *NeuCallbacks
	getNbprops() int
}

//
// objects
//
type NeuNetwork struct {
	cinput    NeuLayerConfig
	chidden   NeuLayerConfig
	coutput   NeuLayerConfig
	tunables  *NeuTunables
	callbacks *NeuCallbacks
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
	// runtime - vectors
	avec   []float64
	zvec   []float64
	deltas []float64
	// runtime - weights, gradients, and 2nd order
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
	// set defaults (unless already defined)
	nn.initunables()
	// algorithm-specific hyper-params
	nn.initgdalg()

	for idx := 0; idx <= nn.lastidx; idx++ {
		layer := nn.layers[idx]
		layer.init(nn)
	}
	if nn.tunables.winit == Xavier {
		nn.initXavier(nil)
	}
	// other useful tunables
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
	// cost
	if nn.tunables.costfunction == nil {
		if len(nn.tunables.costfname) == 0 {
			nn.tunables.costfname = CostMse
		}
		if nn.tunables.costfname == CostMse {
			nn.tunables.costfunction = nn.CostMse
		} else {
			nn.tunables.costfunction = nn.CostCrossEntropy
		}
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
	// nor is it recommended..
	assert(layer.config.actfname != "softmax", "softmax activation for inner layers is currently not supported")

	// init weights
	layer.winit(nn.tunables)

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

func (layer *NeuLayer) winit(tunables *NeuTunables) {
	next := layer.next
	layer.weights = newMatrix(layer.size, next.size, -1.0, 1.0)
	if tunables.winit == Random_1105 {
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				if layer.weights[i][j] >= 0 && layer.weights[i][j] < 0.5 {
					layer.weights[i][j] = 1 - layer.weights[i][j]
				} else if layer.weights[i][j] < 0 && layer.weights[i][j] > -0.5 {
					layer.weights[i][j] = -(1 + layer.weights[i][j])
				}
			}
		}
	}
}

// Xavier et al at http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
func (nn *NeuNetwork) initXavier(newrand *rand.Rand) {
	qsix := math.Sqrt(6)
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		u := qsix / float64(layer.size+next.size)
		d := u * 2
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				if newrand == nil {
					layer.weights[i][j] = d*rand.Float64() - u
				} else {
					layer.weights[i][j] = d*newrand.Float64() - u
				}
			}
		}
	}
}

func (nn *NeuNetwork) sparsify(newrand *rand.Rand, pct int) {
	numweights := 0
	for l := 1; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		numweights += layer.size * next.size
	}
	wm := numweights * pct / 100
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				if newrand == nil {
					k := rand.Intn(numweights)
					if k < wm {
						layer.weights[i][j] = 0
					}
				} else {
					k := newrand.Intn(numweights)
					if k < wm {
						layer.weights[i][j] = 0
					}
				}
			}
		}
	}
}

func (nn *NeuNetwork) reset() {
	nn.initgdalg()
	gdalgname := nn.tunables.gdalgname
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		fillVector(layer.avec, 0.0)
		fillVector(layer.zvec, 0.0)
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

func (nn *NeuNetwork) copyNetwork(from *NeuNetwork, withgrads bool) {
	assert(nn.lastidx == from.lastidx)
	copyStruct(nn.tunables, from.tunables)
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		layer_from := from.layers[l]
		assert(layer.size == layer_from.size)
		assert(layer.next.size == layer_from.next.size)
		copyMatrix(layer.weights, layer_from.weights)

		if withgrads {
			copyMatrix(layer.gradient, layer_from.gradient)
			copyMatrix(layer.pregradient, layer_from.pregradient)
			copyMatrix(layer.rmsgradient, layer_from.rmsgradient)
			if layer.avegradient != nil {
				copyMatrix(layer.avegradient, layer_from.avegradient)
			}
			if layer.rmswupdates != nil {
				copyMatrix(layer.rmswupdates, layer_from.rmswupdates)
			}
			if layer.wupdates != nil {
				copyMatrix(layer.wupdates, layer_from.wupdates)
			}
		}
	}
}

func (nn *NeuNetwork) normalizeY(yvec []float64) []float64 {
	var ynorm = yvec
	cb := nn.getCallbacks()
	if cb != nil && cb.normcbY != nil {
		ynorm = cloneVector(yvec) // FIXME: preallocate
		cb.normcbY(ynorm)
	}
	return ynorm
}

func (nn *NeuNetwork) populateInput(xvec []float64, normalize bool) {
	var xnorm = xvec
	cb := nn.getCallbacks()
	if normalize && cb != nil && cb.normcbX != nil {
		xnorm = cloneVector(xvec)
		cb.normcbX(xnorm)
	}
	ilayer := nn.layers[0]
	copy(ilayer.avec, xnorm) // copies up to the smaller num elems, bias remains intact
}

func (nn *NeuNetwork) forward(xvec []float64) []float64 {
	nn.populateInput(xvec, true)
	return nn.reForward()
}

// feed-forward pass on the already stored input, possibly with different weights
func (nn *NeuNetwork) reForward() []float64 {
	for l := 1; l <= nn.lastidx; l++ {
		nn.forwardLayer(nn.layers[l])
	}
	olayer := nn.layers[nn.lastidx]
	return olayer.avec
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

//
// compute output layer's deltas based on its activation and
// the configured NN's cost function
// NOTE:
// only the most popular (cost, activation) combos are currently supported
//
func (nn *NeuNetwork) computeDeltas(yvec []float64) []float64 {
	assert(len(yvec) == nn.coutput.size)
	olayer := nn.layers[nn.lastidx]
	cname, aname := nn.tunables.costfname, olayer.config.actfname
	assert(cname == CostMse || cname == CostCrossEntropy, "Unexpected cost function")

	actF := activations[aname]
	if cname == CostCrossEntropy && (aname == "sigmoid" || aname == "softmax") {
		for i := 0; i < nn.coutput.size; i++ {
			olayer.deltas[i] = olayer.avec[i] - yvec[i]
		}
	} else {
		for i := 0; i < nn.coutput.size; i++ {
			if cname == CostCrossEntropy {
				if nn.coutput.size == 1 {
					olayer.deltas[i] = (olayer.avec[i] - yvec[i]) / (olayer.avec[i] * (1 - olayer.avec[i]))
				} else {
					olayer.deltas[i] = -yvec[i] / olayer.avec[i]
				}
			} else {
				olayer.deltas[i] = olayer.avec[i] - yvec[i]
			}
			if actF.dfy != nil {
				olayer.deltas[i] *= actF.dfy(olayer.avec[i])
			} else {
				olayer.deltas[i] *= actF.dfx(olayer.zvec[i])
			}
		}
	}
	return olayer.deltas
}

func (nn *NeuNetwork) populateDeltas(deltas []float64) {
	olayer := nn.layers[nn.lastidx]
	copyVector(olayer.deltas, deltas)
}

func (nn *NeuNetwork) backpropDeltas() {
	nn.nbackprops++
	for l := nn.lastidx - 1; l > 0; l-- {
		layer := nn.layers[l]
		next := layer.next
		actF := activations[layer.config.actfname]
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
// recompute weight gradients
// typical sequence: computeDeltas() -- backpropDeltas() -- backpropGradients()
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
			divMatrixNum(layer.gradient, float64(batchsize))
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

//============================================================
//
// training and prediction: externally visible methods
//
//============================================================
func (nn *NeuNetwork) Predict(xvec []float64) []float64 {
	yvec := nn.nnint.forward(xvec)
	var ynorm = yvec
	cb := nn.getCallbacks()
	if cb != nil && cb.denormcbY != nil {
		ynorm = cloneVector(yvec)
		cb.denormcbY(ynorm)
	}
	return ynorm
}

//
// single step in both directions: forward and back
//
func (nn *NeuNetwork) TrainStep(xvec []float64, yvec []float64) {
	nn.nnint.forward(xvec)
	ynorm := nn.nnint.normalizeY(yvec)
	nn.nnint.computeDeltas(ynorm)
	nn.nnint.backpropDeltas()
	nn.nnint.backpropGradients()
}

//============================================================
//
// Get accessors
//
//============================================================
func (nn *NeuNetwork) getIsize() int {
	return nn.cinput.size
}

func (nn *NeuNetwork) getOsize() int {
	return nn.coutput.size
}

func (nn *NeuNetwork) getLayer(l int) *NeuLayer {
	return nn.layers[l]
}

func (nn *NeuNetwork) getTunables() *NeuTunables {
	return nn.tunables
}

func (nn *NeuNetwork) getCallbacks() *NeuCallbacks {
	return nn.callbacks
}

func (nn *NeuNetwork) getHsize() int {
	return nn.chidden.size
}

func (nn *NeuNetwork) getHnum() int {
	return len(nn.layers) - 2
}

func (nn *NeuNetwork) getNbprops() int {
	return nn.nbackprops
}

//============================================================
//
// NeuLayer methods
//
//============================================================
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
		layer.rmsgradient[i][j] = pow2(layer.gradient[i][j])
	} else {
		alpha /= (math.Sqrt(layer.rmsgradient[i][j] + eps))
		layer.rmsgradient[i][j] += pow2(layer.gradient[i][j])
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
		layer.rmsgradient[i][j] = pow2(layer.gradient[i][j])
		deltaij := alpha * layer.gradient[i][j]
		layer.rmswupdates[i][j] = pow2(deltaij)
	} else {
		layer.rmsgradient[i][j] = gamma*layer.rmsgradient[i][j] + g1*pow2(layer.gradient[i][j])
		alpha = math.Sqrt(layer.rmswupdates[i][j]+eps) / (math.Sqrt(layer.rmsgradient[i][j] + eps))
		deltaij := alpha * layer.gradient[i][j]
		layer.rmswupdates[i][j] = gamma*layer.rmswupdates[i][j] + g1*pow2(deltaij)
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
		layer.rmsgradient[i][j] = pow2(layer.gradient[i][j])
	} else {
		layer.rmsgradient[i][j] = gamma*layer.rmsgradient[i][j] + g1*pow2(layer.gradient[i][j])
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
		layer.rmsgradient[i][j] = pow2(layer.gradient[i][j])
		layer.avegradient[i][j] = layer.gradient[i][j]
	} else {
		layer.avegradient[i][j] = nn.tunables.beta1*layer.avegradient[i][j] + (1-nn.tunables.beta1)*layer.gradient[i][j]
		layer.rmsgradient[i][j] = nn.tunables.beta2*layer.rmsgradient[i][j] + (1-nn.tunables.beta2)*pow2(layer.gradient[i][j])
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

//============================================================
//
// NN misc
//
//============================================================
// given an index of the output coord [0, nn.getOsize())
// compute sign vector for the dY(i)/dX(j)
// for all j in the range [0, nn.getIsize())
//
// return true if at least of the partial derivatves is non-zero
func (nn *NeuNetwork) realGradSign(ygsign []float64, yidx int, xvec []float64) (nonzero bool) {
	const eps = GRADCHECK_eps
	const eps2 = eps * 2
	for j := 0; j < nn.getIsize(); j++ {
		xj := xvec[j]
		xvec[j] = xj + eps
		yplus := nn.forward(xvec)[yidx]
		xvec[j] = xj - eps
		yminus := nn.forward(xvec)[yidx]
		xvec[j] = xj
		ygrad := (yplus - yminus) / eps2
		if ygrad > eps2 {
			ygsign[j] = 1
			nonzero = true
		} else if ygrad < -eps2 {
			ygsign[j] = -1
			nonzero = true
		} else {
			ygsign[j] = 0
		}
	}
	return
}

package gorann

import (
	"fmt"
)

//===========================================================================
//
// WeightedMixerNN is a neural network that aggregates
// other neural networks NN1, .. NNn
// with a "mixed" output defined as weighted-sum(NNi)
// This type implements one simple logic to calibrate the mixing weights,
// the logic that is _not_ based on gradient descent..
//
//===========================================================================
type WeightedMixerNN struct {
	NeuNetwork
	nns []NeuNetworkInterface
	// get optimized at rt
	weights []float64
	// work space
	tmpdeltas  []float64
	tmpweights []float64
	tmpcost    []float64
	// heuristrics
	lost          []bool // marks those who'd lost the race and are not running anymore
	scores        []int
	nbpival       int // the following 4 parameters control weight update intervals
	nbpupd        int
	nbpivalcnt    int
	nbpivalcntlim int
	nalive        int // current number of competing nets
	kalive        int // number of NN winners at the end of the race
	// for convenience
	ulayer *NeuLayer
	olayer *NeuLayer
	// config & const
	alpha         float64
	maxcostratio  float64
	maxdeltaratio float64
	epsilon       float64
}

// c-tor
func NewWeightedMixerNN(nns ...NeuNetworkInterface) *WeightedMixerNN {
	assert(len(nns) > 1)
	osize := nns[0].getOsize()
	for i := 1; i < len(nns); i++ {
		assert(nns[0].getIsize() == nns[i].getIsize())
		assert(osize == nns[i].getOsize())
		assert(nns[0].getCallbacks() == nns[i].getCallbacks())
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

	mixer.lost = make([]bool, len(nns))
	mixer.scores = make([]int, len(nns))
	mixer.nalive = len(nns)
	// equal mixing weights
	mixer.weights = newVector(mixer.ulayer.size, 1/float64(len(nns)))

	mixer.callbacks = nns[0].getCallbacks()
	mixer.nnint = mixer

	// constants & defaults
	mixer.alpha = mixer.tunables.alpha
	mixer.maxcostratio = 2
	mixer.maxdeltaratio = 2
	mixer.nbpival = max(cli.nbp, 100000)
	mixer.epsilon = 1E-8

	mixer.nbpivalcntlim = 7
	mixer.kalive = 2
	if mixer.nalive < 8 {
		mixer.kalive = 1
	}
	return mixer
}

//=====================================================================
//
// NeuNetworkInterface methods
//
//=====================================================================

// in the forward pass networks do the work
func (mixer *WeightedMixerNN) forward(xvec []float64) []float64 {
	var xnorm = xvec
	cb := mixer.getCallbacks()
	if cb != nil && cb.normcbX != nil {
		xnorm = cloneVector(xvec)
		cb.normcbX(xnorm)
	}
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].populateInput(xnorm, false)
	}
	return mixer.reForward()
}

// output = weighted-sum(NNi)
func (mixer *WeightedMixerNN) reForward() []float64 {
	osize := mixer.olayer.size
	fillVector(mixer.olayer.avec, 0.0)
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		avec := mixer.nns[i].reForward()
		ii := i * osize
		copy(mixer.ulayer.avec[ii:], avec)
		//
		// ulayer => olayer forward pass (see nn.forwardLayer())
		//
		mulVectorElem(mixer.ulayer.avec[ii:ii+osize], mixer.weights[ii:])
		addVectorElem(mixer.olayer.avec, mixer.ulayer.avec[ii:])
	}
	return mixer.olayer.avec
}

func (mixer *WeightedMixerNN) computeDeltas(yvec []float64) []float64 {
	osize := mixer.olayer.size
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.tmpcost[i] += mixer.nns[i].costfunction(yvec)

		ii := i * osize
		deltas := mixer.nns[i].computeDeltas(yvec)
		copyVector(mixer.tmpdeltas, deltas)

		mulVectorElem(mixer.tmpdeltas, mixer.weights[ii:])
		addVectorElemAbs(mixer.ulayer.deltas[ii:ii+osize], mixer.tmpdeltas)
	}
	return nil
}

func (mixer *WeightedMixerNN) backpropDeltas() {
	mixer.nbackprops++
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].backpropDeltas()
	}
}

func (mixer *WeightedMixerNN) backpropGradients() {
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].backpropGradients()
	}
}

func (mixer *WeightedMixerNN) fixGradients(batchsize int) {
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].fixGradients(batchsize)
	}
}

// NeuNetworkInterface method, uses heuristics
func (mixer *WeightedMixerNN) fixWeights(batchsize int) {
	if mixer.nbackprops-mixer.nbpupd >= mixer.nbpival {
		mixer.fixMixingWeights()
		mixer.zeroSmall()
		mixer.nbpupd = mixer.nbackprops
	}

	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].fixWeights(batchsize)
	}
}

// Get accessor
func (mixer *WeightedMixerNN) getIsize() int {
	return mixer.nns[0].getIsize()
}

//=====================================================================
//
// HEURISTICS: adjust mixing weights
//
//=====================================================================
func (mixer *WeightedMixerNN) fixMixingWin() {
	osize := mixer.olayer.size

	// 1. cost/min-cost
	mincosti := 0
	for i := 1; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		if mixer.tmpcost[i] < mixer.tmpcost[mincosti] {
			mincosti = i
		}
	}
	// 2. min delta wins
	for j := 0; j < osize; j++ {
		mini, maxi := 0, 0
		for i := 1; i < len(mixer.nns); i++ {
			if mixer.lost[i] {
				continue
			}
			if mixer.ulayer.deltas[i*osize+j] < mixer.ulayer.deltas[mini*osize+j] {
				mini = i
			} else if mixer.ulayer.deltas[i*osize+j] > mixer.ulayer.deltas[maxi*osize+j] {
				maxi = i
			}
		}
		for i := 0; i < len(mixer.nns); i++ {
			if mixer.lost[i] {
				continue
			}
			ii := i * osize
			if i == mini && i == mincosti {
				mixer.ulayer.deltas[ii+j] = mixer.alpha
			} else if i == maxi && i != mincosti {
				mixer.ulayer.deltas[ii+j] = -mixer.alpha
			} else {
				mixer.ulayer.deltas[ii+j] = 0
			}
		}
	}
	// 4. adjust mixing weights - deltas
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		ii := i * osize
		addVectorElem(mixer.weights[ii:ii+osize], mixer.ulayer.deltas[ii:ii+osize])
	}

	// 6. finally, rebalance (positive) mixing weights
	mixer.normalizeNegative()
	mixer.rebalancePositive()

	fillVector(mixer.ulayer.deltas, 0.0)
	fillVector(mixer.tmpcost, 0.0)

	mixer.traceWeights()
}

//
// fixMixingWeights() and fixMixingWin() are the two (simple) alternative
// heuristics,
// performing more or less the same in the tests, and either both converge or fail to
//
func (mixer *WeightedMixerNN) fixMixingWeights() {
	osize := mixer.olayer.size

	// 1. cost/min-cost ratios
	mincost := 10000.0
	// fmt.Printf("%d: %.2v\n", mixer.nbackprops, mixer.tmpcost)
	for i := 0; i < len(mixer.nns); i++ {
		mincost = fmin(mincost, mixer.tmpcost[i])
	}
	divVectorNum(mixer.tmpcost, mincost+mixer.epsilon)
	// 2. delta "contributions" by the respective NNs
	copyVector(mixer.tmpdeltas, mixer.ulayer.deltas[:osize])
	for i := 1; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		ii := i * osize
		addVectorElem(mixer.tmpdeltas, mixer.ulayer.deltas[ii:])
	}
	addVectorNum(mixer.tmpdeltas, mixer.epsilon)
	// 3. deltas and delta/min-delta ratios
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		ii := i * osize
		divVectorElemAbs(mixer.ulayer.deltas[ii:ii+osize], mixer.tmpdeltas)
	}
	for j := 0; j < osize; j++ {
		mindelta := 10000.0
		for i := 0; i < len(mixer.nns); i++ {
			if mixer.lost[i] {
				continue
			}
			ii := i * osize
			mindelta = fmin(mindelta, mixer.ulayer.deltas[ii+j])
		}
		for i := 0; i < len(mixer.nns); i++ {
			if mixer.lost[i] {
				continue
			}
			ii := i * osize
			mixer.ulayer.deltas[ii+j] /= mindelta
		}
	}
	// 4. adjust mixing weights - deltas
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
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
		if mixer.lost[i] {
			continue
		}
		if mixer.tmpcost[i] < mixer.maxcostratio {
			ii := i * osize
			for j := 0; j < osize; j++ {
				val := 1 / mixer.tmpcost[i]
				mixer.weights[ii+j] += mixer.alpha * val
			}
		}
	}

	// 6. finally, rebalance mixing weights
	mixer.normalizeNegative()
	mixer.rebalancePositive()

	fillVector(mixer.ulayer.deltas, 0.0)
	fillVector(mixer.tmpcost, 0.0)

	mixer.traceWeights()
}

//
// helpers
//

// makes sure sum(weight_j) == 1 for each j-th neuron
func (mixer *WeightedMixerNN) rebalancePositive() {
	osize := mixer.olayer.size
	fillVector(mixer.tmpweights, 0.0)
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		ii := i * osize
		addVectorElem(mixer.tmpweights, mixer.weights[ii:ii+osize])
	}
	for i := 0; i < mixer.ulayer.size; i++ {
		j := i % osize
		mixer.weights[i] /= mixer.tmpweights[j]
	}
}

func (mixer *WeightedMixerNN) scoreSmall() {
	osize := mixer.olayer.size
	minsum, idx := 1000.0, -1
	if mixer.nalive <= mixer.kalive {
		return
	}
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		ii := i * osize
		sum := 0.0
		for j := 0; j < osize; j++ {
			sum += mixer.weights[ii+j]
		}
		if sum < minsum {
			minsum = sum
			idx = i
		}
	}
	mixer.scores[idx]++
}

func (mixer *WeightedMixerNN) loseSmall(unconditional bool) {
	osize := mixer.olayer.size
	if mixer.nalive <= mixer.kalive {
		return
	}
	sum, maxsc, idx := 0, -1000, -1
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		sum += mixer.scores[i]
		if mixer.scores[i] > maxsc {
			maxsc = mixer.scores[i]
			idx = i
		}
	}
	for i := 0; i < len(mixer.nns); i++ {
		if float64(mixer.scores[i])/float64(sum) > 0.5 {
			mixer.lost[i] = true
			ii := i * osize
			for j := 0; j < osize; j++ {
				mixer.weights[ii+j] = 0
			}
			mixer.nalive--
			unconditional = false
		}
	}
	if unconditional {
		assert(!mixer.lost[idx])
		mixer.lost[idx] = true
		ii := idx * osize
		for j := 0; j < osize; j++ {
			mixer.weights[ii+j] = 0
		}
		mixer.nalive--
	}
	for i := 0; i < len(mixer.nns); i++ {
		mixer.scores[i] = 0
	}
}

func (mixer *WeightedMixerNN) zeroSmall() {
	osize := mixer.olayer.size
	maxsum := -1000.0
	sumvec := newVector(len(mixer.nns))
	if mixer.nalive <= mixer.kalive {
		return
	}
	for i := 0; i < len(mixer.nns); i++ {
		ii := i * osize
		sum, maxelem := 0.0, -1000.0
		for j := 0; j < osize; j++ {
			sum += mixer.weights[ii+j]
			maxelem = fmax(mixer.weights[ii+j], maxelem)
		}
		sum -= maxelem
		maxsum = fmax(sum, maxsum)
		sumvec[i] = sum
	}
	for i := 0; i < len(mixer.nns); i++ {
		sum := sumvec[i]
		if sum > maxsum/20 {
			continue
		}
		mixer.lost[i] = true
		mixer.nalive--
		ii := i * osize
		for j := 0; j < osize; j++ {
			mixer.weights[ii+j] = 0
		}
	}
}

// normalizes the mixing weights into [0.00099, 0.99]
func (mixer *WeightedMixerNN) normalizeNegative() {
	osize := mixer.olayer.size
	for j := 0; j < osize; j++ {
		minweight := 10000.0
		maxweight := -10000.0
		for i := 0; i < len(mixer.nns); i++ {
			if mixer.lost[i] {
				continue
			}
			ii := i * osize
			minweight = fmin(minweight, mixer.weights[ii+j])
			maxweight = fmax(maxweight, mixer.weights[ii+j])
		}
		if maxweight-minweight < mixer.epsilon {
			continue
		}
		for i := 0; i < len(mixer.nns); i++ {
			if mixer.lost[i] {
				continue
			}
			ii := i * osize
			mixer.weights[ii+j] = ((mixer.weights[ii+j]-minweight)/(maxweight-minweight) + 0.001) * 0.99
		}
	}
}

func (mixer *WeightedMixerNN) traceWeights() {
	osize := mixer.olayer.size
	for i := 0; i < len(mixer.nns); i++ {
		ii := i * osize
		fmt.Printf("(%d, %d): %.2v\n", mixer.nns[i].getHsize(), mixer.nns[i].getHnum(), mixer.weights[ii:ii+osize])
	}
}

//===========================================================================
//
// Sub-classing basic NN mixer with the purest GD-based logic
// to calibrate mixing weights
//
//===========================================================================
const (
	matrixToVector = 1
	vectorToMatrix = 2
)

type WeightedGradientNN struct {
	WeightedMixerNN
}

func NewWeightedGradientNN(nns ...NeuNetworkInterface) *WeightedGradientNN {
	weightedmixer := NewWeightedMixerNN(nns...)

	mixer := &WeightedGradientNN{*weightedmixer}
	mixer.tunables.lambda = DEFAULT_lambda

	zeroMatrix(mixer.ulayer.weights)
	mixer.wconvert(vectorToMatrix)
	mixer.nnint = mixer
	return mixer
}

// convert mixer.weights[] <--> mixer.ulayer.weights[][]
func (mixer *WeightedGradientNN) wconvert(dir int) {
	osize := mixer.olayer.size
	for j := 0; j < osize; j++ {
		for i := 0; i < len(mixer.nns); i++ {
			if mixer.lost[i] {
				continue
			}
			ii := i*osize + j
			if dir == vectorToMatrix {
				mixer.ulayer.weights[ii][j] = mixer.weights[ii]
			} else {
				mixer.weights[ii] = mixer.ulayer.weights[ii][j]
			}
		}
	}
}

func (mixer *WeightedGradientNN) computeDeltas(yvec []float64) []float64 {
	mixer.NeuNetwork.computeDeltas(yvec)
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].computeDeltas(yvec)
	}
	return nil
}

func (mixer *WeightedGradientNN) backpropDeltas() {
	mixer.NeuNetwork.backpropDeltas()
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].backpropDeltas()
	}
}

func (mixer *WeightedGradientNN) backpropGradients() {
	osize := mixer.olayer.size
	for i := 0; i < mixer.ulayer.size; i++ {
		for j := 0; j < osize; j++ {
			if i%osize == j {
				mixer.ulayer.gradient[i][j] += mixer.ulayer.avec[i] * mixer.olayer.deltas[j]
			}
		}
	}
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].backpropGradients()
	}
}

func (mixer *WeightedGradientNN) fixGradients(batchsize int) {
	mixer.NeuNetwork.fixGradients(batchsize)
	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].fixGradients(batchsize)
	}
}

// NeuNetworkInterface method, uses heuristics
func (mixer *WeightedGradientNN) fixWeights(batchsize int) {
	mixer.NeuNetwork.fixWeights(batchsize)
	mixer.wconvert(matrixToVector)

	mixer.normalizeNegative()
	mixer.scoreSmall()
	mixer.rebalancePositive()
	mixer.wconvert(vectorToMatrix)

	if mixer.nbackprops-mixer.nbpupd >= mixer.nbpival {
		mixer.nbpivalcnt++
		unconditional := mixer.nbpivalcnt >= mixer.nbpivalcntlim
		mixer.loseSmall(unconditional)
		if unconditional {
			mixer.nbpivalcnt = 0
		}
		mixer.nbpupd = mixer.nbackprops
		mixer.traceWeights()
	}

	for i := 0; i < len(mixer.nns); i++ {
		if mixer.lost[i] {
			continue
		}
		mixer.nns[i].fixWeights(batchsize)
	}
}

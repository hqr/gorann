package gorann

import (
	"math"
	"math/rand"
)

//
// config
//
type EvoTunables struct {
	NeuTunables
	sigma             float64 // sigma aka std
	momentum          float64 // momentum
	hireward, hialpha float64 // high(er) reward threshold and the corresponding learning rate (evolution only)
	rewd              float64 // reward delta - the reward diff btw noise and -noise applied to weights
	nperturb          int     // half the number of the NN weight perturbations aka jitters (evolution only)
	sparsity          int     // jitter sparsity: % weights that get jittered = 100 - sparsity
	jinflate          int     // inflate an already generated gaussian noise (jitter)
	rewdup            int     // reward delta doubling period (the diff between err and -err perturbations)
}

type Evolution struct {
	NeuNetwork
	nn_cpy   *NeuNetwork
	tunables *EvoTunables
	rewards  []float64
	gnoise   [][][][]float64
	newrand  *rand.Rand
	id       int
	// runtime
	nevolves int
	l        int // NN layer round robin
}

func NewEvolution(
	cinput NeuLayerConfig, chidden NeuLayerConfig, numhidden int, coutput NeuLayerConfig, tunables *EvoTunables, id int) (evo *Evolution) {

	nn := NewNeuNetwork(cinput, chidden, numhidden, coutput, &tunables.NeuTunables)
	nn_cpy := NewNeuNetwork(cinput, chidden, numhidden, coutput, &NeuTunables{})
	nn_cpy.copyNetwork(nn)
	gnoise := make([][][][]float64, numhidden+1)
	for l := 0; l < numhidden+1; l++ {
		gnoise[l] = make([][][]float64, tunables.nperturb*2)
		layer := nn.layers[l]
		next := layer.next
		for jj, j := 0, 0; j < tunables.nperturb; j++ {
			gnoise[l][jj] = newMatrix(layer.size, next.size)
			jj++
			gnoise[l][jj] = newMatrix(layer.size, next.size)
			jj++
		}
	}
	rewards := newVector(tunables.nperturb * 2)
	evo = &Evolution{NeuNetwork: *nn, nn_cpy: nn_cpy, tunables: tunables, rewards: rewards, gnoise: gnoise, id: id}

	evo.newrand = rand.New(rand.NewSource(int64((id + 1) * 100)))
	evo.l = evo.newrand.Intn((id + 1) * 100)
	evo.nnint = evo
	return
}

//
// evolve over normally distributed jitter vectors
// - (training) caller is expected to execute the following sequence:
//	computeDeltas(ynorm)
//	backpropDeltas()
//	backpropGradients()
//
func (evo *Evolution) computeDeltas(yvec []float64) []float64 {
	assert(len(yvec) == evo.coutput.size)
	olayer := evo.layers[evo.lastidx]
	copyVector(olayer.deltas, yvec)
	return yvec
}

// overload to generate normal jitter for the layer l
func (evo *Evolution) backpropDeltas() {
	l := evo.l % evo.lastidx
	layer := evo.NeuNetwork.layers[l]
	cols := layer.next.size
	noisycube := evo.gnoise[l]
	for jj, j := 0, 0; j < evo.tunables.nperturb; {
		fillMatrixNormal(noisycube[jj], 0.0, evo.tunables.sigma, evo.tunables.sparsity, evo.newrand)
		copyMatrix(noisycube[jj+1], noisycube[jj])
		mulMatrixNum(noisycube[jj+1], -1.0)
		mat := noisycube[jj]
		assert(len(mat[0]) == cols)
		j++
		jj += 2
		if cols < 4 || cols < evo.tunables.jinflate {
			continue
		}
		// inflate the noise evo.tunables.inflate times
		for i := 0; i < evo.tunables.jinflate-1 && j < evo.tunables.nperturb; i++ {
			reshuffleMatrix(noisycube[jj], mat)
			copyMatrix(noisycube[jj+1], noisycube[jj])
			mulMatrixNum(noisycube[jj+1], -1.0)
			mat = noisycube[jj+1]
			j++
			jj += 2
		}
	}
}

func (evo *Evolution) backpropGradients() {
	evo.nevolves++
	if evo.nevolves%evo.tunables.rewdup == 0 {
		evo.tunables.rewd *= 2
	}
	// round robin: one layer at a time
	l := evo.l % evo.lastidx

	olayer := evo.layers[evo.lastidx]
	y := olayer.deltas[0] // FIXME

	layer, layer_cpy := evo.NeuNetwork.layers[l], evo.nn_cpy.layers[l]
	copyMatrix(layer_cpy.weights, layer.weights)

	noisycube := evo.gnoise[l]
	//
	// estimate the rewards for all 2*nperturb perturbations
	//
	for jj, j := 0, 0; j < evo.tunables.nperturb; j++ {
		addMatrixElem(layer.weights, noisycube[jj])
		avec := evo.reForward()
		evo.rewards[jj] = -math.Pow(avec[0]-y, 2)
		copyMatrix(layer.weights, layer_cpy.weights)
		//
		// and again, this time with a "negative" noise
		//
		addMatrixElem(layer.weights, noisycube[jj+1])
		avec = evo.reForward()
		evo.rewards[jj+1] = -math.Pow(avec[0]-y, 2)
		copyMatrix(layer.weights, layer_cpy.weights)
		jj += 2
	}
	//
	// standardize the rewards and update the gradient
	//
	standardizeVectorZscore(evo.rewards)
	for jj, j := 0, 0; j < evo.tunables.nperturb; j++ {
		if evo.rewards[jj] > evo.rewards[jj+1]+evo.tunables.rewd {
			if evo.rewards[jj] > evo.tunables.hireward {
				mulMatrixNum(noisycube[jj], evo.tunables.hialpha*evo.rewards[jj])
			} else {
				mulMatrixNum(noisycube[jj], evo.tunables.alpha*evo.rewards[jj])
			}
			addMatrixElem(layer.gradient, noisycube[jj])
		} else if evo.rewards[jj+1] > evo.rewards[jj]+evo.tunables.rewd {
			if evo.rewards[jj+1] > evo.tunables.hireward {
				mulMatrixNum(noisycube[jj+1], evo.tunables.hialpha*evo.rewards[jj+1])
			} else {
				mulMatrixNum(noisycube[jj+1], evo.tunables.alpha*evo.rewards[jj+1])
			}
			addMatrixElem(layer.gradient, noisycube[jj+1])
		}
		jj += 2
	}
	evo.l++
}

// average accumulated weight updates over a mini-batch
func (evo *Evolution) fixWeights(batchsize int) {
	nn := &evo.NeuNetwork
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		divMatrixNum(layer.gradient, float64(batchsize))
		addMatrixElem(layer.weights, layer.gradient)

		if evo.tunables.momentum > 0 {
			mulMatrixNum(layer.pregradient, evo.tunables.momentum)
			addMatrixElem(layer.weights, layer.pregradient)
			copyMatrix(layer.pregradient, layer.gradient)
		}
		zeroMatrix(layer.gradient)
	}
}

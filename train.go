package gorann

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

//
// constants
//
const (
	ConvergedCost = 1 << iota
	ConvergedWeight
	ConvergedGradient
	ConvergedMaxBackprops
)

//===========================================================================
//
// Training & Testing Parameters (TTP)
//
//===========================================================================
type TTP struct {
	// the network
	nn *NeuNetwork
	// actual result set, or one of the callbacks below
	resultset   [][]float64
	resultvalcb func(xvec []float64) []float64 // generate yvec for a given xvec
	resultidxcb func(xidx int) []float64       // compute yvec given an index of xvec from the training set
	// repeat training set so many times
	repeat int
	// testing policy
	pct       int  // %% of the training set for testing
	num       int  // number of training instances used for --/--
	randbatch bool // test random batches selected out of the training set
	seqtail   bool // do not use the "tail" of the training set for training, rather use it for testing only
	batchsize int  // based on the size of the training set and the network tunable
	// converged? bitwise sum of conditions below
	// note: Train() routine exits on whatever comes first
	maxweightdelta float64 // L2 norm(previous-weights - current-weights)
	maxgradnorm    float64 // L2 norm(gradient)
	maxcost        float64 // average cost-function(testing-set)
	maxbackprops   int     // max number of back propagations
	// runtime
	weitrack []float64   // L2 norm(previous-weights - current-weights)
	gratrack []float64   // L2 norm(gradient)
	prew     [][]float64 // weights - before
}

// set defaults
func (ttp *TTP) init(m int) {
	assert(ttp.nn != nil)
	assert(ttp.resultset == nil || len(ttp.resultset) == m)
	assert(ttp.resultset != nil || ttp.resultvalcb != nil || ttp.resultidxcb != nil)
	if !ttp.seqtail {
		ttp.randbatch = true
	}
	if ttp.pct == 0 && ttp.num == 0 {
		ttp.pct = 30
	}
	ttp.repeat = max(ttp.repeat, 1)

	ttp.batchsize = ttp.nn.tunables.batchsize
	if ttp.batchsize < 0 || ttp.batchsize > m {
		ttp.batchsize = m
	}
	ttp.weitrack = newVector(10, 1000.0)
	ttp.gratrack = newVector(10, 1000.0)
}

func (ttp *TTP) getYvec(xvec []float64, i int) []float64 {
	var yvec []float64
	switch {
	case ttp.resultset != nil:
		yvec = ttp.resultset[i]
	case ttp.resultvalcb != nil:
		yvec = ttp.resultvalcb(xvec)
	case ttp.resultidxcb != nil:
		yvec = ttp.resultidxcb(i)
	}
	return yvec
}

func (ttp *TTP) gradnormTracker() {
	if ttp.maxgradnorm > 0 {
		nn := ttp.nn
		layer := nn.layers[nn.lastidx-1] // the last hidden layer, next to the output
		edist0 := normL2Matrix(layer.gradient, nil)
		shiftVector(ttp.gratrack)
		pushVector(ttp.gratrack, edist0)
	}
}

func (ttp *TTP) weightdeltaBefore() {
	if ttp.maxweightdelta > 0 {
		nn := ttp.nn
		layer := nn.layers[nn.lastidx-1] // the last hidden layer, next to the output
		ttp.prew = cloneMatrix(layer.weights)
	}
}

func (ttp *TTP) weightdeltaAfter() {
	if ttp.maxweightdelta > 0 {
		nn := ttp.nn
		layer := nn.layers[nn.lastidx-1] // last hidden
		edist := normL2Matrix(ttp.prew, layer.weights)
		shiftVector(ttp.weitrack)
		pushVector(ttp.weitrack, edist)
	}
}

func (ttp *TTP) converged() (cnv int) {
	// num backprops
	if ttp.maxbackprops > 0 && ttp.nn.nbackprops >= ttp.maxbackprops {
		cnv |= ConvergedMaxBackprops
	}
	// weights
	if ltVector(ttp.maxweightdelta, 2, ttp.weitrack) {
		cnv |= ConvergedWeight
	}
	// grads
	if ltVector(ttp.maxgradnorm, 2, ttp.gratrack) {
		cnv |= ConvergedGradient
	}
	return
}

func (ttp *TTP) testRandomBatch(Xs [][]float64, cnv int, nbp int) int {
	if !ttp.randbatch {
		return cnv
	}
	nn, m, num := ttp.nn, len(Xs), ttp.num
	if num == 0 {
		num = m * ttp.pct / 100
	}
	trace_cost := cli.tracenumbp > 0 && cli.tracecost && (nn.nbackprops/cli.tracenumbp > nbp/cli.tracenumbp)
	if ttp.maxcost == 0 && !trace_cost {
		return cnv
	}
	// round up
	numbatches := (num + ttp.batchsize/2) / ttp.batchsize
	numbatches = max(numbatches, 1)
	cost, creg := 0.0, 0.0
	if nn.tunables.lambda > 0 {
		creg = nn.CostL2Regularization()
	}
	for b := 0; b < numbatches; b++ {
		cbatch := 0.0
		for k := 0; k < ttp.batchsize; k++ {
			i := int(rand.Int31n(int32(m)))
			xvec := Xs[i]
			yvec := ttp.getYvec(xvec, i)
			nn.nnint.forward(xvec)
			cbatch += nn.costfunction(yvec)
		}
		cost += (cbatch + creg) / float64(ttp.batchsize)
	}
	if math.Abs(cost) < math.MaxInt16 {
		cost /= float64(numbatches)
		if cost < ttp.maxcost {
			cnv |= ConvergedCost
		}
		if trace_cost {
			log.Print(nn.nbackprops, fmt.Sprintf(" c %f", cost))
		}
	}
	return cnv
}

//===========================================================================
//
// nn training/testing methods
//
//===========================================================================
func (nn *NeuNetwork) Predict(xvec []float64) []float64 {
	yvec := nn.nnint.forward(xvec)
	var ynorm []float64 = yvec
	if nn.callbacks.denormcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.denormcbY(ynorm)
	}
	return ynorm
}

// must be called right after the backprop pass
// (in other words, assumes all the gradients[][][] to be updated)
func (nn *NeuNetwork) CheckGradients(yvec []float64) {
	const eps = GRADCHECK_eps
	const eps2 = eps * 2
	maxdiff, ll, ii, jj, nfail := 0.0, 0, 0, 0, 0
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				w := layer.weights[i][j]
				wplus, wminus := w+eps, w-eps

				layer.weights[i][j] = wplus
				nn.nnint.reForward()
				costplus := nn.costfunction(yvec)

				layer.weights[i][j] = wminus
				nn.nnint.reForward()
				costminus := nn.costfunction(yvec)

				layer.weights[i][j] = w
				// including the cost contributed by regularization
				gradij := (costplus - costminus + nn.costl2regeps(l, i, j, eps)) / eps2

				diff := math.Abs(gradij - layer.gradient[i][j])
				if diff > eps2 {
					nfail++
					if diff > maxdiff {
						maxdiff = diff
						ll, ii, jj = l, i, j

					}
				}
			}
		}

	}
	if nfail > 0 {
		log.Print("grad-check: ", nn.nbackprops, nfail, fmt.Sprintf(" [%2d=>(%2d->%2d)] %.4e", ll, ii, jj, maxdiff))
	}
}

func (nn *NeuNetwork) CheckGradients_Batch(xbatch [][]float64, batchsize int, ttp *TTP, idxbase int) {
	assert(len(xbatch) == batchsize)
	const eps = GRADCHECK_eps
	const eps2 = eps * 2
	maxdiff, ll, ii, jj, nfail := 0.0, 0, 0, 0, 0
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				var costplus, costminus float64
				w := layer.weights[i][j]
				wplus, wminus := w+eps, w-eps
				for k := 0; k < batchsize; k++ {
					xvec := xbatch[k]
					yvec := ttp.getYvec(xvec, idxbase+k)

					layer.weights[i][j] = wplus
					nn.nnint.forward(xvec)
					costplus += nn.costfunction(yvec)

					layer.weights[i][j] = wminus
					nn.nnint.reForward()
					costminus += nn.costfunction(yvec)
				}
				layer.weights[i][j] = w
				// including the cost contributed by L2 regularization
				gradij := (costplus - costminus + nn.costl2regeps(l, i, j, eps)) / float64(batchsize) / eps2
				diff := math.Abs(gradij - layer.gradient[i][j])
				if diff > eps2 {
					nfail++
					if diff > maxdiff {
						maxdiff = diff
						ll, ii, jj = l, i, j

					}
				}
			}
		}

	}
	if nfail > 0 {
		log.Print("grad-batch: ", nn.nbackprops, nfail, fmt.Sprintf(" [%2d=>(%2d->%2d)] %.4e", ll, ii, jj, maxdiff))
	}
}

func (nn *NeuNetwork) TrainStep(xvec []float64, yvec []float64) {
	assert(nn.cinput.size == len(xvec), fmt.Sprintf("num inputs: %d (must be %d)", len(xvec), nn.cinput.size))
	assert(nn.coutput.size == len(yvec), fmt.Sprintf("num outputs: %d (must be %d)", len(yvec), nn.coutput.size))

	nn.nnint.forward(xvec)
	var ynorm []float64 = yvec
	if nn.callbacks.normcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.normcbY(ynorm)
	}
	nn.nnint.backprop(ynorm)
}

func (nn *NeuNetwork) TrainSet(Xs [][]float64, ttp *TTP) int {
	m, bi, batchsize := len(Xs), 0, ttp.batchsize
	for i, xvec := range Xs {
		yvec := ttp.getYvec(xvec, i)
		nn.TrainStep(xvec, yvec)
		bi++
		if bi < batchsize {
			continue
		}
		ttp.gradnormTracker()
		ttp.weightdeltaBefore()
		nn.nnint.fixGradients(batchsize) // <============== (1)
		if cli.checkgrads && (cli.tracenumbp > 0 && nn.nbackprops%cli.tracenumbp == 0) {
			if batchsize == 1 {
				nn.CheckGradients(yvec)
			} else {
				idxbase := i - batchsize + 1
				nn.CheckGradients_Batch(Xs[idxbase:i+1], batchsize, ttp, idxbase)
			}
		}
		nn.nnint.fixWeights(batchsize) // <============== (2)
		ttp.weightdeltaAfter()
		bi = 0
		if batchsize > m-i-1 {
			batchsize = m - i - 1
		}
	}
	return ttp.converged()
}

func (nn *NeuNetwork) Train(Xs [][]float64, ttp *TTP) int {
	converged, nbp := 0, nn.nbackprops
	ttp.init(len(Xs))
	//
	// do the training
	for k := 0; k < ttp.repeat && converged == 0; k++ {
		converged = nn.TrainSet(Xs, ttp)
	}
	// 1. test using a random subset of the training data
	// 2. check convergence on cost
	// 3. trace cost while testing
	return ttp.testRandomBatch(Xs, converged, nbp)
}

func (nn *NeuNetwork) Pretrain(Xs [][]float64, ttp *TTP) {
	nn_orig := NewNeuNetwork(nn.cinput, nn.chidden, nn.lastidx-1, nn.coutput, nn.tunables)
	nn_optm := NewNeuNetwork(nn.cinput, nn.chidden, nn.lastidx-1, nn.coutput, nn.tunables)
	nn_orig.copyNetwork(nn)
	nn_optm.copyNetwork(nn)

	numsteps := 100
	optalgs := []string{"", Adagrad, RMSprop, Adadelta, ADAM}
	alphas := []float64{0.01, 0.05, 0.1}
	gdscopes := []bool{false, true}

	avgcost := float64(math.MaxInt32)
	// prep training and testing sets
	trainingX := newMatrix(numsteps, len(Xs[0]))
	trainingY := newMatrix(numsteps, nn.coutput.size)
	numtesting := numsteps / 3
	testingX := newMatrix(numtesting, len(Xs[0]))
	testingY := newMatrix(numtesting, nn.coutput.size)

	step := 0
	for step < numsteps {
		for i, xvec := range Xs {
			yvec := ttp.getYvec(xvec, i)
			copyVector(trainingX[step], xvec)
			copyVector(trainingY[step], yvec)
			step++
			if step >= numsteps {
				break
			}
		}
	}
	j := 0
	for i := len(Xs) - 1; i >= 0; i-- {
		xvec := Xs[i]
		yvec := ttp.getYvec(xvec, i)
		copyVector(testingX[j], xvec)
		copyVector(testingY[j], yvec)
		j++
		if j >= len(testingX) {
			break
		}
	}
	// cycle through variations
	for _, alg := range optalgs {
		for _, alpha := range alphas {
			for _, scope := range gdscopes {
				// reset nn
				nn.reset()
				nn.copyNetwork(nn_orig)
				// set new tunables and config
				nn.tunables.gdalgname = alg
				nn.initgdalg(alg)
				nn.tunables.alpha = alpha
				nn.tunables.gdalgscopeall = scope
				// use the training set
				nn.Train(trainingX, &TTP{resultset: trainingY})
				// use the testing set to calc the cost
				cost := 0.0
				for i, xvec := range testingX {
					yvec := testingY[i]
					nn.nnint.forward(xvec)
					cost += nn.costfunction(yvec)
				}
				cost /= float64(len(testingX))
				if cost < avgcost {
					avgcost = cost
					nn_optm.copyNetwork(nn)
				}
			}
		}
	}
	// use the best
	nn.reset()
	nn.copyNetwork(nn_optm)
	fmt.Println("pre-train cost:", avgcost)
	fmt.Println("pre-train conf:", nn.tunables)
}

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
	pct int // %% of the training set for testing
	num int // number of training instances used for --/--
	// the following option is useful for time series, for instance
	// instead of selecting random batches out of the training set
	// test cost and check-gradients inline with the training, i. e., sequentially
	sequential bool
	//
	batchsize int // based on the size of the training set and the network tunable
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
	assert(ttp.resultset == nil || len(ttp.resultset) >= m)
	assert(ttp.resultset != nil || ttp.resultvalcb != nil || ttp.resultidxcb != nil)
	if ttp.sequential {
		assert(ttp.repeat < 2, "cannot repeat time series and such")
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
	assert(!ttp.sequential)
	nn, m, num := ttp.nn, len(Xs), ttp.num
	if num == 0 {
		num = m * ttp.pct / 100
	}
	trace_rand_cost := cli.tracecost && (nn.nbackprops/cli.nbp > nbp/cli.nbp)
	if ttp.maxcost == 0 && !trace_rand_cost {
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
		if ttp.maxcost > 0 && cost < ttp.maxcost {
			cnv |= ConvergedCost
		}
		if trace_rand_cost {
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
	var ynorm = yvec
	if nn.callbacks.denormcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.denormcbY(ynorm)
	}
	return ynorm
}

// check gradients for an already back-propagated mini-batch
func (nn *NeuNetwork) CheckGradientsAfterBp(xbatch [][]float64, ttp *TTP, idxbase int) {
	const eps = GRADCHECK_eps
	const eps2 = eps * 2
	maxdiff, ll, ii, jj, nfail, ntotal, batchsize := 0.0, 0, 0, 0, 0, 0, len(xbatch)
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
					if batchsize == 1 {
						nn.nnint.reForward()
					} else {
						nn.nnint.forward(xvec)
					}
					costplus += nn.costfunction(yvec)

					layer.weights[i][j] = wminus
					nn.nnint.reForward()
					costminus += nn.costfunction(yvec)
				}
				layer.weights[i][j] = w
				// including the cost contributed by L2 regularization
				gradij := (costplus - costminus + nn.costl2regeps(l, i, j, eps)) / float64(batchsize) / eps2
				diff := math.Abs(gradij - layer.gradient[i][j])
				ntotal++
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
		log.Print("grad-check: ", nn.nbackprops, fmt.Sprintf(" %d/%d [%2d=>(%2d->%2d)] %.4e", nfail, ntotal, ll, ii, jj, maxdiff))
	}
}

// check gradients for a selected weight inline with processing a given (next) mini-batch
func (nn *NeuNetwork) Train_and_CheckGradients(xbatch [][]float64, ttp *TTP, idxbase int, l, i, j int) {
	const eps = GRADCHECK_eps
	const eps2 = eps * 2
	batchsize, layer, costplus, costminus := len(xbatch), nn.layers[l], 0.0, 0.0
	w := layer.weights[i][j]
	wplus, wminus := w+eps, w-eps
	for k := 0; k < batchsize; k++ {
		xvec := xbatch[k]
		yvec := ttp.getYvec(xvec, idxbase+k)
		nn.TrainStep(xvec, yvec)

		layer.weights[i][j] = wplus
		nn.nnint.reForward()
		costplus += nn.costfunction(yvec)

		layer.weights[i][j] = wminus
		nn.nnint.reForward()
		costminus += nn.costfunction(yvec)
	}
	layer.weights[i][j] = w

	nn.nnint.fixGradients(batchsize) // <============== (1)

	// including the cost contributed by L2 regularization
	gradij := (costplus - costminus + nn.costl2regeps(l, i, j, eps)) / float64(batchsize) / eps2
	diff := math.Abs(gradij - layer.gradient[i][j])
	if diff > eps2 {
		log.Print("grad-batch: ", nn.nbackprops, fmt.Sprintf(" [%2d=>(%2d->%2d)] %.4e", l, i, j, diff))
	}
	nn.nnint.fixWeights(batchsize) // <============== (2)
}

func (nn *NeuNetwork) TrainStep(xvec []float64, yvec []float64) {
	assert(nn.coutput.size == len(yvec), fmt.Sprintf("num outputs: %d (must be %d)", len(yvec), nn.coutput.size))

	nn.nnint.forward(xvec)
	var ynorm = yvec
	if nn.callbacks.normcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.normcbY(ynorm)
	}
	nn.nnint.computeDeltas(ynorm)
	nn.nnint.backpropDeltas()
	nn.nnint.backpropGradients()
}

func (nn *NeuNetwork) TrainSet(Xs [][]float64, ttp *TTP, costnum int) int {
	m, bi, batchsize, cost := len(Xs), 0, ttp.batchsize, 0.0
	for i, xvec := range Xs {
		yvec := ttp.getYvec(xvec, i)
		nn.TrainStep(xvec, yvec)
		// costnum > 0: cumulative cost of the last so-many examples in the set
		if costnum > m-i-1 {
			cost += nn.costfunction(yvec)
		}
		bi++
		if bi < batchsize {
			continue
		}
		ttp.gradnormTracker()
		ttp.weightdeltaBefore()
		nn.nnint.fixGradients(batchsize) // <============== (1)
		if cli.checkgrad && nn.nbackprops%cli.nbp == 0 && (!ttp.sequential || batchsize == 1) {
			idxbase := i - batchsize + 1
			nn.CheckGradientsAfterBp(Xs[idxbase:i+1], ttp, idxbase)
		}
		nn.nnint.fixWeights(batchsize) // <============== (2)
		ttp.weightdeltaAfter()
		bi = 0
		if batchsize > m-i-1 {
			batchsize = m - i - 1
		}
	}
	cnv := ttp.converged()
	if costnum > 0 {
		cost /= float64(costnum)
		if cli.tracecost {
			log.Print(nn.nbackprops, fmt.Sprintf(" c %f", cost))
		}
		if ttp.maxcost > 0 && cost < ttp.maxcost {
			cnv |= ConvergedCost
		}
	}
	return cnv
}

func (nn *NeuNetwork) Train(Xs [][]float64, ttp *TTP) int {
	converged, nbp, m, costnum := 0, nn.nbackprops, len(Xs), 0
	ttp.init(m)
	// case: sequential processing with possible inline tracing and/or grad-checking
	if ttp.sequential {
		if (nn.nbackprops+m)/cli.nbp > nbp/cli.nbp {
			// set 'costnum' to trace cost and/or check cost convergence
			if ttp.maxcost > 0 || cli.tracecost {
				costnum = ttp.num
				if costnum == 0 {
					costnum = m * ttp.pct / 100
				}
			}
		}
		return nn.TrainSet(Xs, ttp, costnum)
	}

	// case: train, possibly multiple times while selecting random batches for cost/grad-checking
	for k := 0; k < ttp.repeat && converged == 0; k++ {
		converged = nn.TrainSet(Xs, ttp, 0)
	}
	// test using a random subset of the already processed training data
	// also, trace cost and check convergence (on cost)
	return ttp.testRandomBatch(Xs, converged, nbp)
}

func (nn *NeuNetwork) Pretrain(Xs [][]float64, ttp *TTP) {
	nn_orig := NewNeuNetwork(nn.cinput, nn.chidden, nn.lastidx-1, nn.coutput, new(NeuTunables))
	nn_optm := NewNeuNetwork(nn.cinput, nn.chidden, nn.lastidx-1, nn.coutput, new(NeuTunables))
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
				nn.initgdalg()
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

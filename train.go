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
	ConvergedMaxBackprops
)

//===========================================================================
//
// stream and its most popular/trivial impl
//
//===========================================================================
type TtpStream interface {
	getXvec(i int) []float64
	getSize() int
	getSidx() int
}

// wraps vanilla array to implement TtpStream interface
type TtpArr [][]float64

func (Xs TtpArr) getXvec(i int) []float64 { return Xs[i] }
func (Xs TtpArr) getSize() int            { return len(Xs) }
func (Xs TtpArr) getSidx() int            { return 0 }

//===========================================================================
//
// Training & Testing Parameters (TTP)
//
//===========================================================================
type TTP struct {
	// the network
	nn NeuNetworkInterface
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
}

// set defaults
func (ttp *TTP) init(m int) {
	assert(ttp.resultset == nil || len(ttp.resultset) >= m)
	assert(ttp.resultset != nil || ttp.resultvalcb != nil || ttp.resultidxcb != nil)
	if ttp.sequential {
		assert(ttp.repeat < 2, "cannot repeat time series and such")
	}
	if ttp.pct == 0 && ttp.num == 0 {
		ttp.pct = 30
	}
	ttp.repeat = max(ttp.repeat, 1)

	ttp.batchsize = ttp.nn.getTunables().batchsize
	if ttp.batchsize <= 0 || ttp.batchsize > m {
		ttp.batchsize = m
	}
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

func (ttp *TTP) convergedBp() (cnv int) {
	if ttp.maxbackprops > 0 && ttp.nn.getNbprops() >= ttp.maxbackprops {
		cnv |= ConvergedMaxBackprops
	}
	return
}

func (ttp *TTP) testRandomBatch(Xs TtpStream, cnv int, nbp int) int {
	assert(!ttp.sequential)
	nn, num := ttp.nn, ttp.num
	m, sidx := Xs.getSize(), Xs.getSidx()
	if num == 0 {
		num = m * ttp.pct / 100
	}
	trace_rand_cost := cli.tracecost && (nn.getNbprops()/cli.nbp > nbp/cli.nbp)
	if ttp.maxcost == 0 && !trace_rand_cost {
		return cnv
	}
	// round up
	numbatches := (num + ttp.batchsize/2) / ttp.batchsize
	numbatches = max(numbatches, 1)
	cost, creg := 0.0, 0.0
	if nn.getTunables().lambda > 0 {
		creg = nn.costl2reg()
	}
	for b := 0; b < numbatches; b++ {
		cbatch := 0.0
		for k := 0; k < ttp.batchsize; k++ {
			i := int(rand.Int31n(int32(m)))
			xvec := Xs.getXvec(sidx + i)
			yvec := ttp.getYvec(xvec, sidx+i)
			nn.forward(xvec)
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
			log.Print(nn.getNbprops(), fmt.Sprintf(" c %f", cost))
		}
	}
	return cnv
}

//===========================================================================
//
// training/testing methods
//
//===========================================================================
// check gradients for an already back-propagated mini-batch
func (ttp *TTP) checkGradientsAfterBp(Xs TtpStream, idxbase int, to int) {
	const eps = GRADCHECK_eps
	const eps2 = eps * 2
	nn := ttp.nn
	maxdiff, ll, ii, jj, nfail, ntotal := 0.0, 0, 0, 0, 0, 0
	batchsize := to - idxbase
	for layer := nn.getLayer(0); layer.next != nil; layer = layer.next {
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				var costplus, costminus float64
				w := layer.weights[i][j]
				wplus, wminus := w+eps, w-eps
				for k := idxbase; k < to; k++ {
					xvec := Xs.getXvec(k)
					yvec := ttp.getYvec(xvec, k)

					layer.weights[i][j] = wplus
					if batchsize == 1 {
						nn.reForward()
					} else {
						nn.forward(xvec)
					}
					costplus += nn.costfunction(yvec)

					layer.weights[i][j] = wminus
					nn.reForward()
					costminus += nn.costfunction(yvec)
				}
				layer.weights[i][j] = w
				// including the cost contributed by L2 regularization
				gradij := (costplus - costminus + nn.costl2_weightij(layer.idx, i, j, eps)) / float64(batchsize) / eps2
				diff := math.Abs(gradij - layer.gradient[i][j])
				ntotal++
				if diff > eps2 {
					nfail++
					if diff > maxdiff {
						maxdiff = diff
						ll, ii, jj = layer.idx, i, j

					}
				}
			}
		}

	}
	if nfail > 0 {
		log.Print("grad-check: ", nn.getNbprops(), fmt.Sprintf(" %d/%d [%2d=>(%2d->%2d)] %.4e", nfail, ntotal, ll, ii, jj, maxdiff))
	}
}

// check gradients for a selected weight inline with processing a given (next) mini-batch
func (ttp *TTP) trainAndCheckGradients(Xs TtpStream, idxbase int, to int, l, i, j int) {
	const eps = GRADCHECK_eps
	const eps2 = eps * 2
	nn := ttp.nn
	batchsize, costplus, costminus := to-idxbase, 0.0, 0.0
	layer := nn.getLayer(l)
	w := layer.weights[i][j]
	wplus, wminus := w+eps, w-eps
	for k := idxbase; k < to; k++ {
		xvec := Xs.getXvec(k)
		yvec := ttp.getYvec(xvec, k)
		nn.TrainStep(xvec, yvec)

		layer.weights[i][j] = wplus
		nn.reForward()
		costplus += nn.costfunction(yvec)

		layer.weights[i][j] = wminus
		nn.reForward()
		costminus += nn.costfunction(yvec)
	}
	layer.weights[i][j] = w

	nn.fixGradients(batchsize) // <============== (1)

	// including the cost contributed by L2 regularization
	gradij := (costplus - costminus + nn.costl2_weightij(l, i, j, eps)) / float64(batchsize) / eps2
	diff := math.Abs(gradij - layer.gradient[i][j])
	if diff > eps2 {
		log.Print("grad-batch: ", nn.getNbprops(), fmt.Sprintf(" [%2d=>(%2d->%2d)] %.4e", l, i, j, diff))
	}
	nn.fixWeights(batchsize) // <============== (2)
}

func (ttp *TTP) TrainSet(Xs TtpStream, costnum int) int {
	sidx, m, bi, batchsize := Xs.getSidx(), Xs.getSize(), 0, ttp.batchsize
	cost := 0.0
	nn := ttp.nn
	for i := sidx; i < sidx+m; i++ {
		xvec := Xs.getXvec(i)
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
		nn.fixGradients(batchsize) // <============== (1)
		if cli.checkgrad && nn.getNbprops()%cli.nbp == 0 && (!ttp.sequential || batchsize == 1) {
			idxbase := i - batchsize + 1
			ttp.checkGradientsAfterBp(Xs, idxbase, i+1)
		}
		nn.fixWeights(batchsize) // <============== (2)
		bi = 0
		if batchsize > m-i-1 {
			batchsize = m - i - 1
		}
	}
	cnv := ttp.convergedBp()
	if costnum > 0 {
		cost /= float64(costnum)
		if cli.tracecost {
			log.Print(nn.getNbprops(), fmt.Sprintf(" c %f", cost))
		}
		if ttp.maxcost > 0 && cost < ttp.maxcost {
			cnv |= ConvergedCost
		}
	}
	return cnv
}

func (ttp *TTP) Train(Xs TtpStream) int {
	nn := ttp.nn
	converged, nbp, m, costnum := 0, nn.getNbprops(), Xs.getSize(), 0
	ttp.init(m)
	// case: sequential processing with possible inline tracing and/or grad-checking
	if ttp.sequential {
		if (nn.getNbprops()+m)/cli.nbp > nbp/cli.nbp {
			// set 'costnum' to trace cost and/or check cost convergence
			if ttp.maxcost > 0 || cli.tracecost {
				costnum = ttp.num
				if costnum == 0 {
					costnum = m * ttp.pct / 100
				}
			}
		}
		return ttp.TrainSet(Xs, costnum)
	}

	// case: train, possibly multiple times while selecting random batches for cost/grad-checking
	for k := 0; k < ttp.repeat && converged == 0; k++ {
		converged = ttp.TrainSet(Xs, 0)
	}
	// test using a random subset of the already processed training data
	// also, trace cost and check convergence (on cost)
	return ttp.testRandomBatch(Xs, converged, nbp)
}

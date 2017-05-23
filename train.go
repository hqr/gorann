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
// training stream and its impl-s
//
//===========================================================================
type TtpStream interface {
	getXvec(i int) []float64
	getYvec(i int) []float64
	getSize() int
	getSidx() int
}

//
// wraps vanilla array to implement TtpStream interface
//
type TtpArr [][]float64

func (Xs TtpArr) getXvec(i int) []float64 { return Xs[i] }
func (Xs TtpArr) getYvec(i int) []float64 {
	assert(false, "undefined for an array - use ttp.result*..")
	return []float64{0}
}
func (Xs TtpArr) getSize() int { return len(Xs) }
func (Xs TtpArr) getSidx() int { return 0 }

//
// rightmost window of a given size "into" a stream
//
type TreamWin struct {
	Xs      [][]float64
	Ys      [][]float64
	leftidx int
	len     int
}

func NewTreamWin(size int, nn NeuNetworkInterface) *TreamWin {
	b := nn.getTunables().batchsize
	assert((size/b)*b == size)

	Xs := newMatrix(size, nn.getIsize())
	Ys := newMatrix(size, nn.getCoutput().size)
	return &TreamWin{Xs: Xs, Ys: Ys}
}
func (trwin *TreamWin) getXvec(i int) []float64 {
	assert(i-trwin.leftidx < trwin.len)
	return trwin.Xs[i-trwin.leftidx]
}
func (trwin *TreamWin) getSize() int { return trwin.len }
func (trwin *TreamWin) getSidx() int { return trwin.leftidx }

func (trwin *TreamWin) addSample(xvec []float64, yvec []float64) {
	if trwin.len < len(trwin.Xs) {
		copyVector(trwin.Xs[trwin.len], xvec)
		copyVector(trwin.Ys[trwin.len], yvec)
		trwin.len++
		return
	}
	copy(trwin.Xs, trwin.Xs[1:])
	copyVector(trwin.Xs[trwin.len-1], xvec)
	copy(trwin.Ys, trwin.Ys[1:])
	copyVector(trwin.Ys[trwin.len-1], yvec)
	trwin.leftidx++

}
func (trwin *TreamWin) getYvec(i int) []float64 {
	assert(i-trwin.leftidx < trwin.len)
	return trwin.Ys[i-trwin.leftidx]
}

//===========================================================================
//
// Training & Testing Parameters (TTP)
//
//===========================================================================
type TTP struct {
	// the network
	nn NeuNetworkInterface
	// the following three fields are used with TtpArr, or to override TtpStream.getYvec()
	resultset    [][]float64
	resultvalcb  func(xvec []float64) []float64 // generate yvec for a given xvec
	resultidxcb  func(xidx int) []float64       // compute yvec given an index of xvec from the training set
	logger       *log.Logger                    //
	maxcost      float64                        // converged |= avgcost < maxcost unless not set
	avgcost      float64                        // runtime
	repeat       int                            // repeat training set so many times
	pct          int                            // %% of the training set for testing
	num          int                            // number of training instances used for --/--
	maxbackprops int                            // max number of back propagations
	batchsize    int                            // based on the size of the training set and the network tunable
	runnerid     int                            //
	sequential   bool                           // random testing is meaningless and is, therefore, forbidden
}

// set defaults
func (ttp *TTP) init(m int) {
	ttp.repeat = max(ttp.repeat, 1)
	assert(!ttp.sequential || ttp.repeat == 1)

	ttp.batchsize = ttp.nn.getTunables().batchsize
	if ttp.batchsize <= 0 || ttp.batchsize > m {
		ttp.batchsize = m
	}
	// avg cost
	if ttp.num == 0 {
		if ttp.pct == 0 {
			ttp.num = ttp.batchsize
			if ttp.num == 1 && m > 50 {
				ttp.num = 10
			}
		} else {
			num := m * ttp.pct / 100
			b := ttp.batchsize
			ttp.num = max((num+b/2)/b*b, b)
		}
	}
}

func (ttp *TTP) getYvec(tstream TtpStream, xvec []float64, i int) []float64 {
	var yvec []float64
	switch {
	case ttp.resultset != nil:
		yvec = ttp.resultset[i]
	case ttp.resultvalcb != nil:
		yvec = ttp.resultvalcb(xvec)
	case ttp.resultidxcb != nil:
		yvec = ttp.resultidxcb(i)
	default:
		yvec = tstream.getYvec(i)
	}
	return yvec
}

//===========================================================================
//
// training/testing methods
//
//===========================================================================
// check gradients for an already back-propagated mini-batch
func (ttp *TTP) checkGradientsAfterBp(tstream TtpStream, idxbase int, to int) {
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
					xvec := tstream.getXvec(k)
					yvec := ttp.getYvec(tstream, xvec, k)

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
		s := fmt.Sprintf(" %d/%d [%2d=>(%2d->%2d)] %.4e", nfail, ntotal, ll, ii, jj, maxdiff)
		if ttp.logger == nil {
			log.Print("grad-check: ", nn.getNbprops(), s)
		} else {
			ttp.logger.Print("grad-check: ", nn.getNbprops(), s)
		}
	}
}

// check gradients for a selected weight inline with processing a given (next) mini-batch
func (ttp *TTP) trainAndCheckGradients(tstream TtpStream, idxbase int, to int, l, i, j int) {
	const eps = GRADCHECK_eps
	const eps2 = eps * 2
	nn := ttp.nn
	batchsize, costplus, costminus := to-idxbase, 0.0, 0.0
	layer := nn.getLayer(l)
	w := layer.weights[i][j]
	wplus, wminus := w+eps, w-eps
	for k := idxbase; k < to; k++ {
		xvec := tstream.getXvec(k)
		yvec := ttp.getYvec(tstream, xvec, k)
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
		s := fmt.Sprintf(" [%2d=>(%2d->%2d)] %.4e", l, i, j, diff)
		if ttp.logger == nil {
			log.Print("grad-batch: ", nn.getNbprops(), s)
		} else {
			ttp.logger.Print("grad-batch: ", nn.getNbprops(), s)
		}
	}
	nn.fixWeights(batchsize) // <============== (2)
}

func (ttp *TTP) TrainSet(tstream TtpStream) int {
	sidx, m, bi, batchsize := tstream.getSidx(), tstream.getSize(), 0, ttp.batchsize
	nn, cost, num := ttp.nn, 0.0, 0
	if ttp.sequential {
		num = ttp.num
	}
	for i := sidx; i < sidx+m; i++ {
		xvec := tstream.getXvec(i)
		yvec := ttp.getYvec(tstream, xvec, i)
		nn.TrainStep(xvec, yvec)
		// cumulative cost of the last so-many examples in the set
		if num > m-(i-sidx)-1 {
			cost += nn.costfunction(yvec)
		}
		bi++
		if bi < batchsize {
			continue
		}
		nn.fixGradients(batchsize) // <============== (1)
		if cli.checkgrad && nn.getNbprops()%cli.nbp == 0 && (!ttp.sequential || batchsize == 1) {
			idxbase := i - batchsize + 1
			ttp.checkGradientsAfterBp(tstream, idxbase, i+1)
		}
		nn.fixWeights(batchsize) // <============== (2)
		bi = 0
		if batchsize > m-i-1 {
			batchsize = m - i - 1
		}
	}
	if num > 0 {
		ttp.avgcost = cost / float64(num)
	}
	if ttp.maxbackprops > 0 && ttp.nn.getNbprops() >= ttp.maxbackprops {
		return ConvergedMaxBackprops
	}
	return 0
}

func (ttp *TTP) costRandomBatch(tstream TtpStream) {
	nn := ttp.nn
	m, sidx := tstream.getSize(), tstream.getSidx()
	ttp.avgcost = 0
	for k := 0; k < ttp.num; k++ {
		i := int(rand.Int31n(int32(m)))
		xvec := tstream.getXvec(sidx + i)
		yvec := ttp.getYvec(tstream, xvec, sidx+i)
		nn.forward(xvec)
		ttp.avgcost += nn.costfunction(yvec)
	}
	ttp.avgcost /= float64(ttp.num)
}

func (ttp *TTP) Train(tstream TtpStream) (cnv int) {
	m := tstream.getSize()
	ttp.init(m)
	nbp := ttp.nn.getNbprops()
	time_to_trace := (nbp+m)/cli.nbp > nbp/cli.nbp
	for k := 0; k < ttp.repeat && cnv == 0; k++ {
		cnv = ttp.TrainSet(tstream)
	}
	if !ttp.sequential {
		ttp.costRandomBatch(tstream)
	}
	if ttp.nn.getTunables().lambda > 0 {
		ttp.avgcost += ttp.nn.costl2reg()
	}
	if ttp.maxcost > 0 && ttp.avgcost < ttp.maxcost {
		cnv |= ConvergedCost
	}
	if time_to_trace && cli.tracecost {
		ttp.logcost()
	}
	return
}

func (ttp *TTP) logcost() {
	nbp := ttp.nn.getNbprops()
	var s string
	if ttp.runnerid == 0 {
		s = fmt.Sprintf("%d: c %f", nbp, ttp.avgcost)
	} else {
		s = fmt.Sprintf("%d: %2d: c %4e", nbp, ttp.runnerid, ttp.avgcost)
	}
	if ttp.logger == nil {
		log.Print(s)
	} else {
		ttp.logger.Print(s)
	}
}

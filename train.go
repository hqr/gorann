package gorann

import (
	"flag"
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

type CommandLine struct {
	tracenumbp int
	tracecost  bool
	checkgrads bool
}

var cli = CommandLine{}

//
// types
//
type TrainParams struct {
	// either actual result set or one of the callbacks
	resultset   [][]float64
	resultvalcb func(xvec []float64) []float64 // generate yvec for a given xvec
	resultidxcb func(xidx int) []float64       // compute yvec given an index of xvec from the training set
	// repeat training set so many times
	repeat int
	// use %% of training set for testing
	testingpct int
	// converged? one or several of the conditions, whatever comes first
	maxweightdelta float64 // L2 norm(previous-weights - current-weights)
	maxgradnorm    float64 // L2 norm(gradient)
	maxcost        float64 // average cost-function(testing-set)
	maxerror       float64 // average L1 norm || yvec - predicted-yvec ||
	maxbackprops   int     // max number of back propagations
}

func init() {
	flag.IntVar(&cli.tracenumbp, "nbp", 0, "trace interval")
	flag.BoolVar(&cli.checkgrads, "grad", false, "check gradients every \"trace interval\"")
	flag.BoolVar(&cli.tracecost, "cost", false, "trace cost every \"trace interval\"")

	flag.Parse()
	assert(!cli.checkgrads || cli.tracenumbp > 0)
	assert(!cli.tracecost || cli.tracenumbp > 0)
}

func (nn *NeuNetwork) Predict(xvec []float64) []float64 {
	yvec := nn.forward(xvec)
	var ynorm []float64 = yvec
	if nn.callbacks.denormcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.denormcbY(ynorm)
	}
	return ynorm
}

// feed-forward pass for the same *already stored* input, possibly with different weights
func (nn *NeuNetwork) reForward() {
	for l := 1; l <= nn.lastidx; l++ {
		nn.forwardLayer(nn.layers[l])
	}
}

// must be called right after the backprop pass
// (in other words, assumes all the gradients[][][] to be updated)
func (nn *NeuNetwork) CheckGradients(yvec []float64) {
	const eps = DEFAULT_eps
	const eps2 = eps * 2
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				layer.weights[i][j] += eps
				nn.reForward()
				costplus := nn.costfunction(yvec)

				layer.weights[i][j] -= eps2
				nn.reForward()
				costminus := nn.costfunction(yvec)

				layer.weights[i][j] += eps // restore
				gradij := (costplus - costminus) / eps2
				diff := math.Abs(gradij - layer.gradient[i][j])
				if diff > eps2 {
					log.Print("grad-check-failed", fmt.Sprintf("[%2d=>(%2d->%2d)] %.4e", l, i, j, diff))
				}
			}
		}

	}
}

func (nn *NeuNetwork) CheckGradients_Batch(xbatch [][]float64, batchsize int, p TrainParams, idxbase int) {
	assert(len(xbatch) == batchsize)
	const eps = DEFAULT_eps
	const eps2 = eps * 2
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				var costplus, costminus float64
				for k := 0; k < batchsize; k++ {
					xvec := xbatch[k]
					yvec := yvecHelper(xvec, idxbase+k, p)

					layer.weights[i][j] += eps
					nn.forward(xvec)
					costplus += nn.costfunction(yvec)

					layer.weights[i][j] -= eps2
					nn.reForward()
					costminus += nn.costfunction(yvec)

					layer.weights[i][j] += eps // restore
				}
				gradij := (costplus - costminus) / float64(batchsize) / eps2
				diff := math.Abs(gradij - layer.gradient[i][j])
				if diff > eps2 {
					log.Print("grad-batch-failed", fmt.Sprintf("[%2d=>(%2d->%2d)] %.4e", l, i, j, diff))
				}
			}
		}

	}
}

func (nn *NeuNetwork) TrainStep(xvec []float64, yvec []float64) {
	assert(nn.cinput.size == len(xvec), fmt.Sprintf("num inputs: %d (must be %d)", len(xvec), nn.cinput.size))
	assert(nn.coutput.size == len(yvec), fmt.Sprintf("num outputs: %d (must be %d)", len(yvec), nn.coutput.size))

	nn.forward(xvec)
	var ynorm []float64 = yvec
	if nn.callbacks.normcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.normcbY(ynorm)
	}
	nn.backprop(ynorm)
}

func (nn *NeuNetwork) TrainSet(Xs [][]float64, p TrainParams, gratrack []float64, weitrack []float64) (converged int) {
	m := len(Xs)
	batchsize, bi := nn.tunables.batchsize, 0
	if batchsize < 0 || batchsize > m {
		batchsize = m
	}
	for i, xvec := range Xs {
		yvec := yvecHelper(xvec, i, p)
		nn.TrainStep(xvec, yvec)
		bi++
		if bi >= batchsize {
			if p.maxgradnorm > 0 {
				nn.gradnormHelper(gratrack)
			}
			var prew [][]float64
			if p.maxweightdelta > 0 {
				prew = nn.weightdeltaBefore()
			}

			nn.fixGradients(batchsize) // <============== (1)
			if cli.checkgrads && (cli.tracenumbp > 0 && nn.nbackprops%cli.tracenumbp == 0) {
				if batchsize == 1 {
					nn.CheckGradients(yvec)
				} else {
					nn.CheckGradients_Batch(Xs[i-batchsize+1:i+1], batchsize, p, i-batchsize+1)
				}
			}
			nn.fixWeights(batchsize) // <============== (2)

			if p.maxweightdelta > 0 {
				nn.weightdeltaAfter(weitrack, prew)
			}
			bi = 0
			if i+batchsize >= m {
				batchsize = m - i - 1
			}
		}
		if p.maxbackprops > 0 && nn.nbackprops >= p.maxbackprops {
			converged |= ConvergedMaxBackprops
			break
		}
	}
	// convergence: weights
	if p.maxweightdelta > 0 {
		i := 0
		for ; i < len(weitrack); i++ {
			if weitrack[i] > p.maxweightdelta {
				break
			}
		}
		if i == len(weitrack) && i > 2 {
			converged |= ConvergedWeight
		}
	}
	// convergence: grads
	if p.maxgradnorm > 0 {
		i := 0
		for ; i < len(gratrack); i++ {
			if gratrack[i] > p.maxgradnorm {
				break
			}
		}
		if i == len(gratrack) && i > 2 {
			converged |= ConvergedGradient
		}
	}
	return
}

func (nn *NeuNetwork) Train(Xs [][]float64, p TrainParams) int {
	m := len(Xs)
	assert(p.resultset == nil || len(p.resultset) == m)
	assert(p.resultset != nil || p.resultvalcb != nil || p.resultidxcb != nil)

	var weitrack []float64 = newVector(10) // L2 norm(previous-weights - current-weights)
	var gratrack []float64 = newVector(10) // L2 norm(gradient)
	repeat := max(p.repeat, 1)
	converged := 0
	nbp := nn.nbackprops
	// do the training
	for k := 0; k < repeat && converged == 0; k++ {
		converged = nn.TrainSet(Xs, p, gratrack, weitrack)
	}
	// 1. test using a random subset of the training data
	// 2. check convergence on cost
	// 3. trace cost while testing
	trace_cost := cli.tracenumbp > 0 && cli.tracecost && (nn.nbackprops/cli.tracenumbp > nbp/cli.tracenumbp)
	if p.maxcost > 0 || trace_cost {
		testingpct := p.testingpct
		if testingpct == 0 {
			testingpct = 30
		}
		testingnum := len(Xs) * testingpct / 100
		cost := 0.0
		for k := 0; k < testingnum; k++ {
			i := int(rand.Int31n(int32(m)))
			xvec := Xs[i]
			yvec := yvecHelper(xvec, i, p)
			nn.forward(xvec)
			cost += nn.costfunction(yvec)
		}
		if testingnum > 0 && math.Abs(cost) < math.MaxInt16 {
			cost /= float64(testingnum)
			if cost <= p.maxcost {
				converged |= ConvergedCost
			}
			if trace_cost {
				log.Print(nn.nbackprops, fmt.Sprintf(" c %f", cost))
			}
		}
	}
	return converged
}

func (nn *NeuNetwork) gradnormHelper(gratrack []float64) {
	layer := nn.layers[nn.lastidx-1] // the last hidden layer, next to the output
	edist0 := normL2Matrix(layer.gradient, nil)
	shiftVector(gratrack)
	pushVector(gratrack, edist0)
}

func (nn *NeuNetwork) weightdeltaBefore() [][]float64 {
	layer := nn.layers[nn.lastidx-1] // the last hidden layer, next to the output
	prew := cloneMatrix(layer.weights)
	return prew
}

func (nn *NeuNetwork) weightdeltaAfter(weitrack []float64, prew [][]float64) {
	layer := nn.layers[nn.lastidx-1] // last hidden
	edist := normL2Matrix(prew, layer.weights)
	shiftVector(weitrack)
	pushVector(weitrack, edist)
}

func yvecHelper(xvec []float64, i int, p TrainParams) []float64 {
	var yvec []float64
	switch {
	case p.resultset != nil:
		yvec = p.resultset[i]
	case p.resultvalcb != nil:
		yvec = p.resultvalcb(xvec)
	case p.resultidxcb != nil:
		yvec = p.resultidxcb(i)
	}
	return yvec
}

func (nn *NeuNetwork) Pretrain(Xs [][]float64, p TrainParams) {
	assert(p.resultset != nil || p.resultvalcb != nil || p.resultidxcb != nil)
	nn_orig := NewNeuNetwork(nn.cinput, nn.chidden, nn.lastidx-1, nn.coutput, nn.tunables.gdalgname)
	nn_optm := NewNeuNetwork(nn.cinput, nn.chidden, nn.lastidx-1, nn.coutput, nn.tunables.gdalgname)
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
			yvec := yvecHelper(xvec, i, p)
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
		yvec := yvecHelper(xvec, i, p)
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
				nn.Train(trainingX, TrainParams{resultset: trainingY})
				// use the testing set to calc the cost
				cost := 0.0
				for i, xvec := range testingX {
					yvec := testingY[i]
					nn.forward(xvec)
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

// debug
func logWei(iter int, weitrack []float64) {
	var w string = " w"
	for j := 0; j < len(weitrack); j++ {
		if weitrack[j] == 0 {
			continue
		}
		w += fmt.Sprintf(" %.6f", weitrack[j])
	}
	log.Print(iter, w)
}

func logGra(iter int, gratrack []float64) {
	var g string = " g"
	for j := 0; j < len(gratrack); j++ {
		if gratrack[j] == 0 {
			continue
		}
		g += fmt.Sprintf(" %.6f", gratrack[j])
	}
	log.Print(iter, g)
}

func max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

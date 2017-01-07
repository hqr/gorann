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
	ConvergedMaxIterations
	ConvergedMaxBackprops
)

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
	// max counts
	maxiterations int // max iterations on the given trainging set
	maxbackprops  int // max total number of back propagations
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

func (nn *NeuNetwork) Train(Xs [][]float64, p TrainParams) int {
	m := len(Xs)
	assert(p.resultset == nil || len(p.resultset) == m)
	assert(p.resultset != nil || p.resultvalcb != nil || p.resultidxcb != nil)
	batchsize := nn.tunables.batchsize
	if batchsize < 0 || batchsize > m {
		batchsize = m
	}
	bi, repeat, converged, iter := 0, p.repeat, 0, 0
	if repeat <= 0 {
		repeat = 1
	}
	// the last so-many L2 norm(previous-weights - current-weights) and L2norm(gradient)
	var weitrack []float64 = newVector(10)
	var gratrack []float64 = newVector(10)
	var nw, ng int
Loop:
	for k := 0; k < repeat; k++ {
		for i, xvec := range Xs {
			yvec := yvecHelper(xvec, i, p)
			nn.TrainStep(xvec, yvec)
			if p.maxgradnorm > 0 {
				nn.gradnormHelper(gratrack)
				ng++
			}
			bi++
			if bi >= batchsize {
				var prew [][]float64
				if p.maxweightdelta > 0 {
					prew = nn.weightdeltaBefore()
					nw++
				}
				nn.fixWeights(batchsize)
				if p.maxweightdelta > 0 {
					nn.weightdeltaAfter(weitrack, prew)
				}
				bi = 0
				if i+batchsize >= m && k == repeat-1 {
					batchsize = m - i - 1
				}
			}
			iter++
			if p.maxiterations > 0 && iter >= p.maxiterations {
				converged |= ConvergedMaxIterations
				break Loop
			}
			if p.maxbackprops > 0 && nn.nbackprops >= p.maxbackprops {
				converged |= ConvergedMaxBackprops
				break Loop
			}
		}
	}
	// convergence: max cost
	if p.maxcost > 0 {
		testingpct := p.testingpct
		if testingpct == 0 {
			testingpct = 10
		}
		testingnum := len(Xs) * testingpct / 100
		if testingnum <= 2 {
			testingnum = 2
		}
		cost := 0.0
		for k := 0; k < testingnum; k++ {
			i := int(rand.Int31n(int32(m)))
			xvec := Xs[i]
			yvec := yvecHelper(xvec, i, p)
			nn.forward(xvec)
			if nn.tunables.costfunction == CostLogistic {
				cost += nn.CostLogistic(yvec)
			} else {
				cost += nn.CostLinear(yvec)
			}
		}
		cost /= float64(testingnum)
		if cost <= p.maxcost {
			converged |= ConvergedCost
		}
	}
	// convergence: weights
	if p.maxweightdelta > 0 && nw > 3 {
		failed := false
		for i := 0; i < len(weitrack); i++ {
			if weitrack[i] > p.maxweightdelta {
				failed = true
			}
		}
		if !failed {
			converged |= ConvergedWeight
		}
	}
	// convergence: grads
	if p.maxgradnorm > 0 && ng > 3 {
		failed := false
		for i := 0; i < len(gratrack); i++ {
			if gratrack[i] > p.maxgradnorm {
				failed = true
			}
		}
		if !failed {
			converged |= ConvergedGradient
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
	nn_orig.copyNetwork(nn, true)
	nn_optm.copyNetwork(nn, true)

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
				nn.copyNetwork(nn_orig, false)
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
					if nn.tunables.costfunction == CostLogistic {
						cost += nn.CostLogistic(yvec)
					} else {
						cost += nn.CostLinear(yvec)
					}
				}
				cost /= float64(len(testingX))
				if cost < avgcost {
					avgcost = cost
					nn_optm.copyNetwork(nn, false)
				}
			}
		}
	}
	// use the best
	nn.copyNetwork(nn_optm, false)
	fmt.Println("pre-train cost:", avgcost)
	fmt.Println("pre-train conf:", nn.tunables)
}

// debug
func logWei(iter int, weitrack []float64) {
	var w string
	for j := 0; j < len(weitrack); j++ {
		if weitrack[j] == 0 {
			continue
		}
		w += fmt.Sprintf(" %.3f", weitrack[j])
	}
	log.Print(iter, w)
}
func logGra(iter int, gratrack []float64) {
	var g string
	for j := 0; j < len(gratrack); j++ {
		if gratrack[j] == 0 {
			continue
		}
		g += fmt.Sprintf(" %.3f", gratrack[j])
	}
	log.Print(iter, g)
}

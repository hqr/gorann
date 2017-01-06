package gorann

import (
	"fmt"
	"math"
	"math/rand"
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

func (nn *NeuNetwork) Train(Xs [][]float64, p TrainParams) bool {
	m := len(Xs)
	assert(p.resultset == nil || len(p.resultset) == m)
	assert(p.resultset != nil || p.resultvalcb != nil || p.resultidxcb != nil)
	batchsize := nn.tunables.batchsize
	if batchsize < 0 || batchsize > m {
		batchsize = m
	}
	bi := 0
	repeat := p.repeat
	if repeat <= 0 {
		repeat = 1
	}
	iter := 0
Loop:
	for k := 0; k < repeat; k++ {
		for i, xvec := range Xs {
			yvec := yvecHelper(xvec, i, p)
			nn.TrainStep(xvec, yvec)
			bi++
			if bi >= batchsize {
				nn.fixWeights(batchsize)
				bi = 0
				if i+batchsize >= m && k == repeat-1 {
					batchsize = m - i - 1
				}
			}
			iter++
			if p.maxiterations > 0 && iter >= p.maxiterations {
				break Loop
			}
			if p.maxbackprops > 0 && nn.nbackprops >= p.maxbackprops {
				break Loop
			}
		}
	}
	// convergence
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
		return cost < p.maxcost
	}

	return false
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
func (nn *NeuNetwork) printTracks(iter int) {
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		var w, g string
		for j := 0; j < len(layer.weitrack); j++ {
			w += fmt.Sprintf(" %.3f", layer.weitrack[j])
		}
		for j := 0; j < len(layer.gratrack); j++ {
			g += fmt.Sprintf(" %.3f", layer.gratrack[j])
		}
		fmt.Printf("%6d: [%2d L2(w changes)]%s\n", iter, layer.idx, w)
		fmt.Printf("%6d: [%2d L2(gradients)]%s\n", iter, layer.idx, g)
	}
}

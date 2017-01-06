package gorann

import (
	"fmt"
	"math"
)

func (nn *NeuNetwork) Predict(xvec []float64) []float64 {
	yvec := nn.forward(xvec)
	var ynorm []float64 = yvec
	if nn.callbacks.denormcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.denormcbY(ynorm)
	}
	return ynorm
}

func (nn *NeuNetwork) TrainStep(xvec []float64, yvec []float64, fy func([]float64) []float64) {
	assert(nn.cinput.size == len(xvec), "wrong num Xs")

	nn.forward(xvec)
	if fy != nil {
		yvec = fy(xvec)
	}
	assert(nn.coutput.size == len(yvec), "wrong num Ys")
	var ynorm []float64 = yvec
	if nn.callbacks.normcbY != nil {
		ynorm = cloneVector(yvec)
		nn.callbacks.normcbY(ynorm)
	}
	nn.backprop(ynorm)
}

func (nn *NeuNetwork) Train(Xs [][]float64, arg interface{}) {
	var Ys [][]float64
	var fy func([]float64) []float64
	m := len(Xs)
	batchsize := nn.tunables.batchsize
	if batchsize < 0 || batchsize > m {
		batchsize = m
	}
	switch arg.(type) {
	case [][]float64:
		Ys = arg.([][]float64)
		assert(m == len(Ys))
	case func([]float64) []float64:
		fy = arg.(func([]float64) []float64)
	}
	// do train
	bi := 0
	for i, xvec := range Xs {
		var yvec []float64
		if Ys != nil {
			yvec = Ys[i]
		}
		assert(yvec == nil || fy == nil)
		nn.TrainStep(xvec, yvec, fy)
		bi++
		if bi >= batchsize {
			nn.fixWeights(batchsize)
			bi = 0
			if i+batchsize >= m {
				batchsize = m - i - 1
			}
		}
	}
}

func (nn *NeuNetwork) Pretrain(Xs [][]float64, arg interface{}) {
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
	var Ys [][]float64
	var fy func([]float64) []float64
	var yvec []float64

	switch arg.(type) {
	case [][]float64:
		Ys = arg.([][]float64)
	case func([]float64) []float64:
		fy = arg.(func([]float64) []float64)
	}
	step := 0
	for step < numsteps {
		for i, xvec := range Xs {
			if fy == nil {
				yvec = Ys[i]
			} else {
				yvec = fy(xvec)
			}
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
		if fy == nil {
			yvec = Ys[i]
		} else {
			yvec = fy(xvec)
		}
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
				nn.Train(trainingX, trainingY)
				// use the testing set to calc the cost
				cost := 0.0
				for i, xvec := range testingX {
					yvec = testingY[i]
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

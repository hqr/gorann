package gorann

import (
	"math"
)

func (nn *NeuNetwork) costfunction(yvec []float64) (cost float64) {
	if nn.tunables.costfname == CostCrossEntropy {
		cost = nn.CostCrossEntropy(yvec)
	} else {
		cost = nn.CostMse(yvec)
	}
	return
}

//
// cost helpers: the yvec is true (training) value
// must be called after the corresponding feed-forward step
//
func (nn *NeuNetwork) CostMse(yvec []float64) float64 {
	assert(len(yvec) == nn.coutput.size)
	ynorm := nn.normalizeY(yvec)
	olayer := nn.layers[nn.lastidx]
	return normL2VectorSquared(ynorm, olayer.avec) / 2
}

// the "L2 regularization" part of the cost
func (nn *NeuNetwork) costl2reg() (creg float64) {
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		for i := 0; i < layer.config.size; i++ { // excepting bias
			for j := 0; j < next.size; j++ {
				creg += pow2(layer.weights[i][j])
			}
		}
	}
	creg = nn.tunables.lambda * creg / 2
	return
}

func (nn *NeuNetwork) costl2_weightij(l int, i int, j int, eps float64) float64 {
	layer := nn.layers[l]
	if nn.tunables.lambda == 0 || i >= layer.config.size {
		return 0
	}
	wij := layer.weights[i][j]
	return nn.tunables.lambda * (pow2(wij+eps) - pow2(wij-eps)) / 2
}

func (nn *NeuNetwork) CostCrossEntropy(yvec []float64) float64 {
	assert(len(yvec) == nn.coutput.size)
	ynorm := nn.normalizeY(yvec)
	olayer := nn.layers[nn.lastidx]
	aname, avec, zvec := olayer.config.actfname, olayer.avec, olayer.zvec
	var err float64

	// special case: binary classification
	if len(ynorm) == 1 {
		// special sub-case: logistic regression
		if aname == "sigmoid" {
			err = ynorm[0]*zvec[0] - math.Log(1+math.Exp(zvec[0]))
			return -err
		}
		ynorm = []float64{ynorm[0], 1 - ynorm[0]}
		avec = []float64{avec[0], 1 - avec[0]}
	}
	for i := 0; i < len(ynorm); i++ {
		err += ynorm[i] * math.Log(avec[i])
	}
	return -err
}

// unlike the conventional Cost functions above, this one works on denormalized result vectors
// note also that it returns L1 norm
func (nn *NeuNetwork) AbsError(yvec []float64) float64 {
	olayer := nn.layers[nn.lastidx]
	var ydenorm = olayer.avec
	cb := nn.getCallbacks()
	if cb != nil && cb.denormcbY != nil {
		ydenorm = cloneVector(olayer.avec)
		cb.denormcbY(ydenorm)
	}
	return normL1Vector(ydenorm, yvec)
}

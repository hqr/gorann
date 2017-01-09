package gorann

import "math"

//
// types
//
type Activation struct {
	name string
	f    func(float64) float64
	dfy  func(float64) float64 // used when the derivative can be expressed as a function of the value of the function
	dfx  func(float64) float64
}

//
// static
//
var activations map[string]*Activation

func init() {
	RegisterActivation(&Activation{"sigmoid", sigmoid, dsigmoid, nil})
	RegisterActivation(&Activation{"tanh", tanh, dtanh, nil})
	RegisterActivation(&Activation{"relu", relu, nil, drelu})
	RegisterActivation(&Activation{"leakyrelu", leakyrelu, nil, dleakyrelu})
	RegisterActivation(&Activation{"softplus", softplus, nil, dsoftplus})
}

func RegisterActivation(a *Activation) {
	if activations == nil {
		activations = make(map[string]*Activation, 8)
	}
	activations[a.name] = a
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}

func tanh(x float64) float64 {
	y1 := math.Exp(x)
	y2 := math.Exp(-x)
	return (y1 - y2) / (y1 + y2)
}

func dtanh(y float64) float64 {
	return 1 - y*y
}

func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func drelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// aka parametric Rectifier
const LeakyApha = 1E-4

func leakyrelu(x float64) float64 {
	if x < 0 {
		return LeakyApha * x
	}
	return x
}

func dleakyrelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return LeakyApha
}

func softplus(x float64) float64 {
	return math.Log(1 + math.Exp(x))
}

func dsoftplus(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

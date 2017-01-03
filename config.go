package gorann

type NeuLayerConfig struct {
	actfname string // activation function (non-linearity) used for a given layer
	size     int    // number of nodes ("neurons") comprising the layer
}

type NeuCallbacks struct {
	normcbX   func(vec []float64)
	normcbY   func(vec []float64)
	denormcbY func(vec []float64)
}

// global configuration and per network instance defaults
const (
	// gradient descent: batching and mini-batching
	BatchSGD         = 1
	BatchTrainingSet = -1
	//
	// regularization ("bias vs. overfitting")
	//
	Lambda       = 0.0
	RegularizeL1 = 1 << 0
	RegularizeL2 = 1 << 1
	//
	// gradient descent optimization algorithms (http://sebastianruder.com/optimizing-gradient-descent)
	//
	Adagrad                = "Adagrad"
	Adadelta               = "Adadelta"
	RMSprop                = "RMSprop"
	GDoptimizationScopeAll = 1 << 0
	//
	// cost function
	//
	CostLinear   = "LMS"      // least mean squares
	CostLogistic = "Logistic" // -y*log(h) - (1-y)*log(1-h)
	//
	// hyperparameters (https://en.wikipedia.org/wiki/Hyperparameter_optimization)
	//
	Epsilon = 0.00001
	Gamma   = 0.9
)

// neural network tunables
type NeuTunables struct {
	alpha          float64 // learning rate, typically 0.01 through 0.1
	momentum       float64 // https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
	costfunction   string  // squared euclidean distance aka LMS | Logistic
	gdalgname      string  // gradient descent optimization algorithm (see above)
	gdalgscope     int     // whether to apply optimization algorithm to all layers or just the last one
	batchsize      int     // gradient descent: BatchSGD | BatchTrainingSet | minibatch
	regularization int
}

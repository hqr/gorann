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
	// gradient descent non-linear optimization algorithms (e.g., http://sebastianruder.com/optimizing-gradient-descent)
	// TODO: Nesterov, ADAM, BFGS, and LBFGS
	//
	Adagrad                = "Adagrad"
	Adadelta               = "Adadelta"
	RMSprop                = "RMSprop"
	ADAM                   = "ADAM"
	GDoptimizationScopeAll = 1 << 0
	//
	// cost function
	//
	CostLinear   = "LMS"      // least mean squares
	CostLogistic = "Logistic" // -y*log(h) - (1-y)*log(1-h)
	//
	// runtime tracking (weight, gradient, etc.) changes - can be used to control the behavior
	//
	TrackWeightChanges   = 1 << 0
	TrackCostChanges     = 1 << 1
	TrackGradientChanges = 1 << 2
)

//
// hyperparameters used with a given opt alg (https://en.wikipedia.org/wiki/Hyperparameter_optimization)
//
// ADAM as per https://arxiv.org/pdf/1412.6980.pdf
const (
	ADAM_alpha   = 0.001
	ADAM_beta1   = 0.9
	ADAM_beta2   = 0.999
	ADAM_beta1_t = 1
	ADAM_beta2_t = 1
	ADAM_eps     = 1E-8
)

// Defaults across all algorithms
const DEFAULT_batchsize = BatchSGD
const DEFAULT_alpha = 0.01
const DEFAULT_momentum = 0.6

// Defaults for: Adagrad, Adadelta, RMSprop
const GDALG_eps = 1E-5
const GDALG_gamma = 0.9

// neural network tunables - a superset
type NeuTunables struct {
	alpha          float64 // learning rate, typically 0.01 through 0.1
	gdalgalpha     float64 // learning rate that is preferred with a specific optimization
	momentum       float64 // https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
	beta1          float64
	beta2          float64
	beta1_t        float64
	beta2_t        float64
	gamma          float64
	eps            float64
	costfunction   string // half squared euclidean distance (L2 norm) aka LMS | Logistic
	gdalgname      string // gradient descent optimization algorithm (see above)
	gdalgscope     int    // whether to apply optimization algorithm to all layers or just the last one
	batchsize      int    // gradient descent: BatchSGD | BatchTrainingSet | minibatch
	regularization int
	tracking       int
}

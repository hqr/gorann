package gorann

import (
	"flag"
)

type NeuLayerConfig struct {
	actfname string // activation function (non-linearity) used for a given layer
	size     int    // number of nodes ("neurons") comprising the layer
}

type NeuCallbacks struct {
	normcbX   func(vec []float64)
	normcbY   func(vec []float64)
	denormcbY func(vec []float64)
}

// gradient descent enums
const (
	BatchSGD         = 1
	BatchTrainingSet = -1
	//
	// gradient descent non-linear optimization algorithms (e.g., http://sebastianruder.com/optimizing-gradient-descent)
	// TODO: conjugate gradients (https://en.wikipedia.org/wiki/Conjugate_gradient_method), BFGS, and LBFGS
	//
	Adagrad  = "Adagrad"
	Adadelta = "Adadelta"
	RMSprop  = "RMSprop"
	ADAM     = "ADAM"
	Rprop    = "Rprop"
	//
	// cost function
	//
	CostMse          = "MSE"          // mean squared error
	CostCrossEntropy = "CrossEntropy" // -(y*log(h) + (1-y)*log(1-h))
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
	Rprop_eta    = 1.2
	Rprop_neta   = 0.5
)

// default hyper-parameters
const (
	DEFAULT_batchsize = BatchSGD
	DEFAULT_alpha     = 0.01 // learning rate
	DEFAULT_momentum  = 0.6  // momentum
	DEFAULT_lambda    = 0.01 // regularization
	// Adagrad, Adadelta, RMSprop
	DEFAULT_eps   = 1E-5
	GRADCHECK_eps = 1E-4
	DEFAULT_gamma = 0.9
)

// neural network tunables - a superset
type NeuTunables struct {
	// hyperparameters used by various gradient descent alg-s
	alpha      float64 // learning rate, typically 0.01 through 0.1
	gdalgalpha float64 // learning rate that is preferred with a specific optimization
	momentum   float64 // https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
	beta1      float64
	beta2      float64
	beta1_t    float64
	beta2_t    float64
	gamma      float64
	eps        float64
	eta        float64
	neta       float64
	lambda     float64 // regularization lambda (default = 0)
	// other config
	costfname     string // half squared euclidean distance (L2 norm) aka LMS | Logistic
	gdalgname     string // gradient descent optimization algorithm (see above)
	gdalgscopeall bool   // whether to apply optimization algorithm to all layers or just the one facing output
	batchsize     int    // gradient descent: BatchSGD | BatchTrainingSet | minibatch
}

//
// CLI
//
type CommandLine struct {
	// override nn props hardcoded via the corresponding c-tors
	// value 0 (zero) or "" is interpreted as not-set
	alpha     float64
	momentum  float64
	lambda    float64
	gdalgname string
	batchsize int
	// tracing and logging
	nbp       int
	tracecost bool
	checkgrad bool
	// FIXME, remove
	lessrnn int
}

var cli = CommandLine{}

func init() {
	flag.Float64Var(&cli.alpha, "alpha", 0, "learning rate: controls the rate at which current changes influence next-updates")
	flag.Float64Var(&cli.momentum, "momentum", 0, "controls the second part in next-update formula: current-change + momentum * previous-update")
	flag.Float64Var(&cli.lambda, "lambda", 0, "controls regularization term in the cost function")
	flag.StringVar(&cli.gdalgname, "gdalgname", "", "optimization algorithm: [ Adagrad | Adadelta | RMSprop | ADAM | Rprop ]")
	flag.IntVar(&cli.batchsize, "batchsize", 0, "as in \"mini-batch gradient descent\"")

	flag.IntVar(&cli.nbp, "nbp", 100000, "trace interval: the number of back propagations (default 100000)")
	flag.BoolVar(&cli.checkgrad, "checkgrad", false, "check gradients every \"trace interval\"")
	flag.BoolVar(&cli.tracecost, "tracecost", false, "trace cost every \"trace interval\"")

	flag.IntVar(&cli.lessrnn, "lessrnn", 0, "... lesser RNN ...")

	flag.Parse()
}

func Configure() {
}

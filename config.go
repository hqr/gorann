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

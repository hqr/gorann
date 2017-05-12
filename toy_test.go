//
// courtesy of [https://blog.openai.com/evolution-strategies]
//
package gorann

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func fn_nespy(w []float64, solution []float64) float64 {
	// reward = -np.sum(np.square(solution - w))
	return -normL2VectorSquared(solution, w)
}

func Test_nespy(t *testing.T) {
	rand.Seed(0)
	// hyperparameters
	var npop = 50     // population size
	var sigma = 0.1   // noise standard deviation
	var alpha = 0.001 // learning rate

	// start the optimization
	var solution = []float64{0.5, 0.1, -0.3}

	w := newVector(3, 0.0, 1.0, "normal")
	for i := 0; i < 300; i++ {
		if i%20 == 0 {
			fmt.Printf("%d, %.5v, %.5f\n", i, w, fn_nespy(w, solution))
		}

		N := newMatrix(npop, 3, 0.0, 1.0, "normal")
		R := newVector(npop)

		for j := 0; j < npop; j++ {
			// w_try := w + sigma*N[j]
			nj := cloneVector(N[j])
			mulVectorNum(nj, sigma)
			w_try := cloneVector(w)
			addVectorElem(w_try, nj)
			R[j] = fn_nespy(w_try, solution) // evaluate the jittered version
		}

		// standardize the rewards to have a gaussian distribution
		// A = (R - np.mean(R)) / np.std(R)
		A := cloneVector(R)
		standardizeVectorZscore(A)
		// perform the parameter update. The matrix multiply below
		// is just an efficient way to sum up all the rows of the noise matrix N,
		// where each row N[j] is weighted by A[j]
		// w = w + alpha/(npop*sigma) * np.dot(N.T, A)
		lrate := alpha / (float64(npop) * sigma)
		for j := 0; j < npop; j++ {
			mulVectorNum(N[j], lrate*A[j])
			addVectorElem(w, N[j])
		}
	}
}

func Test_nespy_alt(t *testing.T) {
	rand.Seed(0)
	// hyperparameters
	var npop = 50     // population size
	var sigma = 0.1   // noise standard deviation
	var alpha = 0.001 // learning rate

	// start the optimization
	var solution = []float64{0.5, 0.1, -0.3}
	var rThresh = 10.0
	w := newVector(3, 0.0, 1.0, "normal")
	for i := 0; i < 600; i++ {
		if i%20 == 0 {
			fmt.Printf("%d, %.5v, %.5f\n", i, w, fn_nespy(w, solution))
		}

		N := newMatrix(npop, 3, 0.0, 1.0, "normal")
		R := newVector(npop)

		for j := 0; j < npop; j++ {
			// w_try := w + sigma*N[j]
			mulVectorNum(N[j], sigma)
			w_try := cloneVector(w)
			addVectorElem(w_try, N[j])
			R[j] = fn_nespy(w_try, solution) // evaluate the jittered version
		}

		// standardize the rewards to have a gaussian distribution
		// A = (R - np.mean(R)) / np.std(R)
		A := cloneVector(R)
		mean, std := standardizeVectorZscore(A)
		for j := 0; j < npop; j++ {
			// prioritize those parameter updates that yield bigger rewards
			if A[j] > mean+rThresh*std {
				mulVectorNum(N[j], 0.1*A[j])
				addVectorElem(w, N[j])
				rThresh += 0.5
			} else {
				mulVectorNum(N[j], alpha*A[j])
				addVectorElem(w, N[j])
			}
		}
	}
}

//==================================================================================
//
// https://rmcantin.bitbucket.io/html/demos.html#hart6func
//
//==================================================================================
var hart6_mA = [][]float64{
	{10.0, 3.0, 17.0, 3.5, 1.7, 8.0},
	{0.05, 10.0, 17.0, 0.1, 8.0, 14.0},
	{3.0, 3.5, 1.7, 10.0, 17.0, 8.0},
	{17.0, 8.0, 0.05, 10.0, 0.1, 14.0}}

var hart6_mC = []float64{1.0, 1.2, 3.0, 3.2}

var hart6_mP = [][]float64{
	{0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886},
	{0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991},
	{0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650},
	{0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381}}

var hart6_opt = []float64{0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573}

var hart6_nn *NeuNetwork
var hart6_newrand *rand.Rand
var hart6_cpy = []float64{0, 0, 0, 0, 0, 0}

func fn_hartmann(xvec []float64) (yvec []float64) {
	yvec = []float64{0}
	for i := 0; i < 4; i++ {
		sum := 0.0
		for j := 0; j < 6; j++ {
			val := xvec[j] - hart6_mP[i][j]
			sum -= hart6_mA[i][j] * val * val
		}
		yvec[0] -= hart6_mC[i] * math.Exp(sum)
	}
	yvec[0] /= -3.33 // NOTE: normalize into (0, 1)
	return
}

func hart6_fill(Xs [][]float64) {
	for i := 0; i < len(Xs); i++ {
		prevmax := -1.0
		for j := 0; j < 10; j++ {
			fillVector(hart6_cpy, 0.0, 1.0, "", hart6_newrand)
			y := hart6_nn.nnint.forward(hart6_cpy)
			if y[0] > prevmax {
				copy(Xs[i], hart6_cpy)
				prevmax = y[0]
			}
		}
	}
}

func Test_hartmann(t *testing.T) {
	/*xvec := newVector(6)
	for i := 0; i < 1E7; i++ {
		fillVector(xvec, 0.0, 1.0)
		y := fn_hartmann(xvec)
		if y[0] > 1 {
			fmt.Printf("%.3f: %.5f\n", y, xvec)
		}
	}
	fmt.Printf("optimum : %.5f\n", fn_hartmann(hart6_opt))*/
	input := NeuLayerConfig{size: 6}
	hidden := NeuLayerConfig{"tanh", input.size * 3}
	output := NeuLayerConfig{"sigmoid", 1}
	tu := &NeuTunables{gdalgname: ADAM, batchsize: 10, winit: XavierNewRand}
	etu := &EvoTunables{
		NeuTunables: *tu,
		sigma:       0.1,           // normal-noise(0, sigma)
		momentum:    0.01,          //
		hireward:    5.0,           // diff (in num stds) that warrants a higher learning rate
		hialpha:     0.01,          //
		rewd:        0.001,         // FIXME: consider using reward / 1000
		rewdup:      rclr.coperiod, // reward delta doubling period
		nperturb:    64,            // half the number of the NN weight perturbations aka jitters
		sparsity:    10,            // noise matrix sparsity (%)
		jinflate:    2}             // gaussian noise inflation ratio, to speed up evolution

	evo := NewEvolution(input, hidden, 3, output, etu, 0)
	nn := &evo.NeuNetwork
	hart6_nn = &evo.NeuNetwork
	hart6_newrand = evo.newrand
	evo.initXavier(hart6_newrand)

	evo.nnint = nn // NOTE: comment this statement, to compare versus Evolution

	//
	// henceforth, the usual training (regression) on random samples
	//
	Xs := newMatrix(10000, nn.nnint.getIsize())
	converged := 0
	prevmax := 0.0
	ttp := &TTP{nn: nn, resultvalcb: fn_hartmann, pct: 10, maxbackprops: 1E7, repeat: 3}
	for converged == 0 {
		hart6_fill(Xs)
		converged = nn.Train(Xs, ttp)

		for i := 0; i < len(Xs); i++ {
			avec := nn.nnint.forward(hart6_opt)
			if avec[0] > prevmax {
				fmt.Printf("%.5f : %.5f\n", hart6_opt, avec[0])
				prevmax = avec[0]
			}
		}
	}
	var mse float64
	// try the trained network on some new data
	hart6_fill(Xs)
	num := 8
	for k := 0; k < num; k++ {
		i := int(rand.Int31n(int32(len(Xs))))
		xvec := Xs[i]
		yvec := fn_hartmann(xvec)
		nn.nnint.forward(xvec)
		mse += nn.costfunction(yvec)
	}
	fmt.Printf("mse %.5f\n", mse/float64(num))
}

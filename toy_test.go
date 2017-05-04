//
// courtesy of [https://blog.openai.com/evolution-strategies]
//
package gorann

import (
	"fmt"
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

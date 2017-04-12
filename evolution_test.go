//
// courtesy of [https://blog.openai.com/evolution-strategies]
//
package gorann

import (
	"fmt"
	"math/rand"
	"testing"
)

func f(w []float64) float64 {
	// reward = -np.sum(np.square(solution - w))
	return -normL2VectorSquared(solution, w)
}

// hyperparameters
var npop = 50     // population size
var sigma = 0.1   // noise standard deviation
var alpha = 0.001 // learning rate

// start the optimization
var solution = []float64{0.5, 0.1, -0.3}

func Test_nespy(t *testing.T) {
	rand.Seed(0)

	w := newVector(3, 0.0, 1.0, "normal")
	for i := 0; i < 300; i++ {
		if i%20 == 0 {
			fmt.Printf("%d, %v, %.5f\n", i, w, f(w))
		}

		N := newMatrix(npop, 3, 0.0, 1.0, "normal")
		R := newVector(npop)

		for j := 0; j < npop; j++ {
			// w_try := w + sigma*N[j]
			nj := cloneVector(N[j])
			mulVectorNum(nj, sigma)
			w_try := cloneVector(w)
			addVectorElem(w_try, nj)
			R[j] = f(w_try) // evaluate the jittered version
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

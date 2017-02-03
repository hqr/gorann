package gorann

import (
	"fmt"
	// "github.com/gonum/matrix/mat64" // pretty print only
	"log"
	"math"
	"math/rand"
	"reflect"
)

func assert(cond bool, args ...interface{}) {
	if cond {
		return
	}
	var message = "assertion failed"
	if len(args) > 0 {
		message += ": "
		for i := 0; i < len(args); i++ {
			message += fmt.Sprintf("%#v ", args[i])
		}
	}
	log.Panic(message)
}

func newMatrix(rows, cols int, args ...interface{}) [][]float64 {
	m := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		if len(args) == 0 {
			m[i] = newVector(cols)
		} else if len(args) == 1 {
			m[i] = newVector(cols, args[0])
		} else if len(args) == 2 {
			m[i] = newVector(cols, args[0], args[1])
		}
	}
	return m
}

func cloneMatrix(src [][]float64) [][]float64 {
	rows := len(src)
	dst := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		dst[r] = newVector(len(src[r]))
		copy(dst[r], src[r])
	}
	return dst
}

func copyMatrix(dst [][]float64, src [][]float64) {
	rows := len(src)
	assert(rows == len(dst))
	cols := len(src[0])
	for r := 0; r < rows; r++ {
		assert(cols == len(dst[r]))
		copy(dst[r], src[r])
	}
}

func zeroMatrix(mat [][]float64) {
	rows := len(mat)
	for r := 0; r < rows; r++ {
		fillVector(mat[r], 0)
	}
}

func mulColVec(mat [][]float64, c int, vec []float64, count int) float64 {
	var sum float64
	for r := 0; r < count; r++ {
		sum += mat[r][c] * vec[r]
	}
	return sum
}

func mulRowVec(mat [][]float64, r int, vec []float64, count int) float64 {
	var sum float64
	for c := 0; c < count; c++ {
		sum += mat[r][c] * vec[c]
	}
	return sum
}

/* - works; commented out for now
func ppMatrix(name string, mat [][]float64) {
	rows := len(mat)
	cols := len(mat[0])
	dense := mat64.NewDense(rows, cols, make([]float64, rows*cols))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			dense.Set(r, c, mat[r][c])
		}
	}
	fa := mat64.Formatted(dense, mat64.Prefix("        "))
	fmt.Printf("%s = %.2f\n\n", name, fa)
}
*/

func divElemMatrix(mat [][]float64, d float64) {
	for r := 0; r < len(mat); r++ {
		row := mat[r]
		divElemVector(row, d)
	}
}

// this is Frobenius norm (not to confuse with L2,1 norm)
func normL2Matrix(matA [][]float64, matB [][]float64) float64 {
	rows := len(matA)
	assert(matB == nil || rows == len(matB))
	edist := 0.0
	for r := 0; r < rows; r++ {
		if matB == nil {
			edist += normL2VectorSquared(matA[r], nil)
		} else {
			edist += normL2VectorSquared(matA[r], matB[r])
		}
	}
	return math.Sqrt(edist)
}

func normL2VectorSquared(avec []float64, bvec []float64) float64 {
	cols := len(avec)
	assert(bvec == nil || cols == len(bvec))
	edist := 0.0
	for c := 0; c < cols; c++ {
		if bvec == nil {
			edist += math.Pow(avec[c], 2)
		} else {
			edist += math.Pow(avec[c]-bvec[c], 2)
		}
	}
	return edist
}

func normL1Vector(avec []float64, bvec []float64) float64 {
	cols := len(avec)
	assert(bvec == nil || cols == len(bvec))
	l1norm := 0.0
	for c := 0; c < cols; c++ {
		if bvec == nil {
			l1norm += math.Abs(avec[c])
		} else {
			l1norm += math.Abs(avec[c] - bvec[c])
		}
	}
	return l1norm
}

func divElemVector(vec []float64, d float64) {
	for c := 0; c < len(vec); c++ {
		vec[c] /= d
	}
}

func mulElemVector(vec []float64, d float64) {
	for c := 0; c < len(vec); c++ {
		vec[c] *= d
	}
}

func newVector(size int, args ...interface{}) []float64 {
	v := make([]float64, size)
	if len(args) == 0 {
		return v
	}
	if len(args) == 1 {
		f := args[0].(float64)
		for i := 0; i < size; i++ {
			v[i] = f
		}
		return v
	}
	// fill in with random values between spec-ed boundaries
	assert(len(args) == 2)
	left := args[0].(float64)
	right := args[1].(float64)
	assert(right > left)
	d := right - left
	for i := 0; i < size; i++ {
		v[i] = d*rand.Float64() + left
	}
	return v
}

func fillVector(vec []float64, x float64) {
	for i := 0; i < len(vec); i++ {
		vec[i] = x
	}
}

func copyVector(dst []float64, src []float64) {
	assert(len(dst) == len(src))
	copy(dst, src)
}

func cloneVector(src []float64) []float64 {
	var dst []float64 = make([]float64, len(src))
	copy(dst, src)
	return dst
}

func shiftVector(vec []float64) {
	copy(vec, vec[1:])
}

func pushVector(vec []float64, x float64) {
	vec[len(vec)-1] = x
}

func ltVector(max float64, num int, vec []float64) bool {
	if max > 0 && len(vec) > num {
		i := 0
		for ; i < len(vec); i++ {
			if vec[i] >= max {
				return false
			}
		}
		return i > num
	}
	return false
}

//
// misc
//
func copyStruct(dst interface{}, src interface{}) {
	x := reflect.ValueOf(src)
	if x.Kind() == reflect.Ptr {
		starX := x.Elem()
		y := reflect.New(starX.Type())
		starY := y.Elem()
		starY.Set(starX)
		reflect.ValueOf(dst).Elem().Set(y.Elem())
	} else {
		dst = x.Interface()
	}
}

func max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

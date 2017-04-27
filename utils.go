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
		m[i] = newVector(cols, args...)
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

func addMatrixElem(dst [][]float64, src [][]float64) {
	rows := len(src)
	assert(rows == len(dst))
	for r := 0; r < rows; r++ {
		addVectorElem(dst[r], src[r])
	}
}

func zeroMatrix(mat [][]float64) {
	rows := len(mat)
	for r := 0; r < rows; r++ {
		fillVector(mat[r], 0)
	}
}

func fillMatrixNormal(mat [][]float64, mean, std float64, skip int, sparsity int) {
	rows := len(mat)
	for r := 0; r < rows; r++ {
		fillVectorNormal(mat[r], mean, std, skip, sparsity)
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

/*
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

func mulMatrixNum(mat [][]float64, d float64) {
	for r := 0; r < len(mat); r++ {
		row := mat[r]
		mulVectorNum(row, d)
	}
}

func divMatrixNum(mat [][]float64, d float64) {
	for r := 0; r < len(mat); r++ {
		row := mat[r]
		divVectorNum(row, d)
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

func meanVector(vec []float64) (mean float64) {
	flen := float64(len(vec))
	for c := 0; c < len(vec); c++ {
		mean += vec[c]
	}
	mean /= flen
	return
}
func meanStdVector(vec []float64) (mean, std float64) {
	flen := float64(len(vec))
	mean = meanVector(vec)
	cpy := cloneVector(vec)
	addVectorNum(cpy, -mean)
	std = math.Sqrt(normL2VectorSquared(cpy, nil) / flen)
	return
}

// z-score standardization
func standardizeVectorZscore(vec []float64) {
	flen := float64(len(vec))
	mean := meanVector(vec)
	addVectorNum(vec, -mean)
	std := math.Sqrt(normL2VectorSquared(vec, nil) / flen)
	divVectorNum(vec, std)
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

// divide vector by a number
func divVectorNum(vec []float64, d float64) {
	for c := 0; c < len(vec); c++ {
		vec[c] /= d
	}
}

// element-wise division: dst /= src
func divVectorElem(dst []float64, src []float64) {
	for c := 0; c < len(dst); c++ {
		dst[c] /= src[c]
	}
}

// element-wise division absolute: dst = math.Abs(dst/src)
func divVectorElemAbs(dst []float64, src []float64) {
	for c := 0; c < len(dst); c++ {
		dst[c] = math.Abs(dst[c]) / math.Abs(src[c])
	}
}

// multiply vector by a number
func mulVectorNum(vec []float64, d float64) {
	for c := 0; c < len(vec); c++ {
		vec[c] *= d
	}
}

// element-wise product: dst *= src
func mulVectorElem(dst []float64, src []float64) {
	for c := 0; c < len(dst); c++ {
		dst[c] *= src[c]
	}
}

// add number to a vector
func addVectorNum(vec []float64, d float64) {
	for c := 0; c < len(vec); c++ {
		vec[c] += d
	}
}

// element-wise add: dst += src
func addVectorElem(dst []float64, src []float64) {
	for c := 0; c < len(dst); c++ {
		dst[c] += src[c]
	}
}

// element-wise add absolute values: dst += math.Abs(src)
func addVectorElemAbs(dst []float64, src []float64) {
	for c := 0; c < len(dst); c++ {
		dst[c] += math.Abs(src[c])
	}
}

// element-wise subtraction: dst -= src
func subVectorElem(dst []float64, src []float64) {
	for c := 0; c < len(dst); c++ {
		dst[c] -= src[c]
	}
}

// element-wise dst = math.Abs(dst - src)
func subVectorElemAbs(dst []float64, src []float64) {
	for c := 0; c < len(dst); c++ {
		dst[c] = math.Abs(dst[c] - src[c])
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
	if len(args) == 2 {
		left := args[0].(float64)
		right := args[1].(float64)
		assert(right > left)
		d := right - left
		for i := 0; i < size; i++ {
			v[i] = d*rand.Float64() + left
		}
		return v
	}

	assert(len(args) == 3)
	dist := args[2].(string)
	assert(dist == "normal")
	mean := args[0].(float64)
	std := args[1].(float64)
	for i := 0; i < size; i++ {
		v[i] = rand.NormFloat64()*std + mean
	}
	return v
}

func fillVector(vec []float64, x float64) {
	for i := 0; i < len(vec); i++ {
		vec[i] = x
	}
}

func fillVectorNormal(vec []float64, mean, std float64, skip int, sparsity int) {
	for i := 0; i < skip; i++ {
		_ = rand.NormFloat64()
	}
	for i := 0; i < len(vec); i++ {
		if sparsity == 0 || rand.Intn(100) >= sparsity {
			vec[i] = rand.NormFloat64()*std + mean
		} else {
			vec[i] = mean
		}
	}
}

func copyVector(dst []float64, src []float64) {
	assert(len(dst) == len(src))
	copy(dst, src)
}

func cloneVector(src []float64) []float64 {
	var dst = make([]float64, len(src))
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
func fmax(a float64, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
func fmin(a float64, b float64) float64 {
	if a > b {
		return b
	}
	return a
}

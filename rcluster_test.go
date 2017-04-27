package gorann

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// types
type Initiator struct {
	nn      *NeuNetwork
	nn_cpy  *NeuNetwork
	reptvec []*Target
	// selected target, its estimated latency, and a copy of (delayed) history
	target    *Target
	minlat    float64
	distory   []float64 // [[selectcnt / ni]]
	gnoise    [][][][]float64
	score     int
	id        int
	genjitter bool
}

type Target struct {
	history   []float64 // [[selectcnt / ni]]
	currlat   float64
	selectcnt int
	id        int
}

// static
var ivec []*Initiator
var tvec []*Target
var rclr *RCLR

//
// config
//
type RCLR struct {
	wg                     *sync.WaitGroup
	mstart                 *sync.RWMutex
	mend                   *sync.RWMutex
	sigma                  float64 // sigma aka std
	rewardelta             float64 // reward delta
	momentum               float64 // momentum
	highreward, hialpha    float64 // high(er) reward threshold and the corresponding learning rate (evolution only)
	svcrate                float64 // service rate
	scorelow, scorehigh    float64 // scoring thresholds that control inter-initiator weights exchange
	ni, nt, nsteps, copies int     // numbers of initiators and targets, number of steps (epochs), number of replicas
	hisize                 int     // target (selection) history size
	coperiod               int     // colaboration = scoring period: number of steps (epochs)
	idelay, tdelay         int     // initiator and target "delay" as far as the history of the target /selections/
	nperturb               int     // half the number of the NN weight /perturbations/ (evolution only)
	sparsity               int     // jitter sparsity: percentage of the weights that get jittered = 100 - sparsity
	batchsize, numhid      int     // NN batchsize, NN number of hidden layers
	gdalg                  string  // NN gradient descent algorithm
	// flags
	useGD       bool // evolution or else (GD)
	repeatevo   bool // reuse a given mini-batch to /evolve/ one more time
	usemomentum bool // apply the previous update * momentum
	reusenoise  bool // reshuffle (and reuse) previously generated noise
	//
	fnlatency func(h []float64) float64
}

func init() {
	rclr = &RCLR{
		&sync.WaitGroup{},
		&sync.RWMutex{},
		&sync.RWMutex{},
		0.5,       //         gaussian sigma
		0.001,     //         rewardelta
		0.01,      //         momentum (evolution)
		0.9, 0.01, //         high(er) reward threshold and the corresponding learning rate (evolution only)
		1.1,         //       service rate: num requests target can handle during one epoch step
		1000.1, 1.3, //       low(TODO) and high scoring thresholds, to control inter-initiator weight exchange
		10, 10, 5000, 3, //   numbers of initiators and targets, num steps, number of replicas
		30,          //       target history size - selection counts that each target keeps back in time
		100,         //       colaboration = scoring period: number of steps (epochs)
		0,           //       (0, 1) => initiator and target see the same exact history
		1,           //       (1, 0) => initiator's history is two epochs older than the target's
		32,          //       half the number of the NN weight /perturbations/ (evolution only)
		0,           //       jitter sparsity
		10, 2, ADAM, //       NN: batchsize, number of hidden layers, GD optimization algorithm
		// flags
		false, // true - use GD, false - run evolution
		true,  // repeat: re-evolve using the same mini-batch
		true,  // use momentum
		false, // reshuffle (and reuse) previously generated noise
		//
		fn4_latency, //       the latency function
	}
	ivec = make([]*Initiator, rclr.ni)
	tvec = make([]*Target, rclr.nt)
}

//
// main
//
func Test_rcluster(t *testing.T) {
	rand.Seed(0)
	construct_T()
	construct_I()

	rclr.mstart.Lock()
	rclr.wg.Add(rclr.ni)
	assert(rclr.coperiod%rclr.batchsize == 0)
	for _, initiator := range ivec {
		if rclr.useGD {
			go initiator.gradientDescent()
		} else {
			go initiator.evolution()
		}
	}

	for step := 1; step <= rclr.nsteps; step++ {
		compute_I() // I: select 3 rep-holding Ts, compute latencies based on (delayed) uvecs, select the T min
		compute_T() // T: given the selections (see prev step), compute "true" latencies and append to the uvecs
		score_I()   // I: +1 if selected the fastest target out of 3

		// synchronization block
		// fmt.Printf("========= %d =========\n", step)
		rclr.mend.Lock()
		rclr.mstart.Unlock()
		rclr.wg.Wait() // ===============> evolution OR gradient descent happens here
		rclr.mstart.Lock()
		rclr.wg.Add(rclr.ni)
		rclr.mend.Unlock()

		rejitter_I(step)

		// inter-initiator weights exchange & printouts
		if step%rclr.coperiod == 0 {
			collab_I() // I: try to make use of the NN(s) with better results

			C, S := newVector(rclr.ni), newVector(rclr.ni)
			for i, initiator := range ivec {
				C[i] = math.Pow(initiator.minlat-initiator.target.currlat, 2)
				S[i] = float64(initiator.score) / float64(rclr.coperiod)

				// zero-out the scores for the next coperiod
				initiator.score = 0
			}
			// L := newVector(rclr.nt)
			// for i, target := range tvec {
			// 	L[i] = target.currlat
			// }
			// fmt.Printf("C: %.3f\n", C)
			fmt.Printf("%3d: %.3f\n", step/rclr.coperiod, S)
			// fmt.Printf("L: %.3f\n\n", L)

			rclr.rewardelta *= 2 // FIXME
		}
	}
}

func construct_T() {
	for i, _ := range tvec {
		tvec[i] = &Target{history: newVector(rclr.hisize + rclr.idelay + rclr.tdelay + 1), id: i}
	}
}

func construct_I() {
	for i, _ := range ivec {
		// multilayer NN is good to approximate about anything
		input := NeuLayerConfig{size: rclr.hisize}
		hidden := NeuLayerConfig{"sigmoid", input.size * 2}
		output := NeuLayerConfig{"identity", 1}
		tunables := &NeuTunables{gdalgname: rclr.gdalg, batchsize: rclr.batchsize, winit: Xavier}

		nn := NewNeuNetwork(input, hidden, rclr.numhid, output, tunables)
		nn_cpy := NewNeuNetwork(input, hidden, rclr.numhid, output, &NeuTunables{})
		nn_cpy.copyNetwork(nn)

		initiator := &Initiator{
			nn:        nn,
			nn_cpy:    nn_cpy,
			reptvec:   make([]*Target, rclr.copies),
			distory:   newVector(rclr.hisize),
			id:        i,
			genjitter: true}

		if !rclr.useGD {
			initiator.gnoise = make([][][][]float64, rclr.numhid+1)
			for l := 0; l < rclr.numhid+1; l++ {
				initiator.gnoise[l] = make([][][]float64, rclr.nperturb*2)
				layer := initiator.nn.layers[l]
				next := layer.next
				for jj, j := 0, 0; j < rclr.nperturb; j++ {
					initiator.gnoise[l][jj] = newMatrix(layer.size, next.size)
					jj++
					initiator.gnoise[l][jj] = newMatrix(layer.size, next.size)
					jj++
				}
			}
		}
		ivec[i] = initiator
	}
}

func compute_I() {
	for _, target := range tvec {
		target.selectcnt = 0
	}
	for _, initiator := range ivec {
		// 3 targets holding the replica that we want to read
		for i := 0; i < rclr.copies; i++ {
			initiator.reptvec[i] = tvec[rand.Intn(rclr.nt)]
		}

		// estimate the respective latencies, select T min
		initiator.target = nil
		for i := 0; i < rclr.copies; i++ {
			xvec := initiator.reptvec[i].history[rclr.idelay : rclr.idelay+rclr.hisize]
			assert(len(xvec) == rclr.hisize)
			avec := initiator.nn.Predict(xvec)
			if initiator.target == nil || initiator.minlat > avec[0] {
				initiator.minlat = avec[0]
				initiator.target = initiator.reptvec[i]
				copy(initiator.distory, xvec)
			}
		}
		initiator.target.selectcnt++
	}
}

func compute_T() {
	for _, target := range tvec {
		// make place for the new entry in history
		//
		copy(target.history[1:], target.history)
		target.history[0] = float64(target.selectcnt) / float64(rclr.ni)
		//
		// target's latency is defined by its (disk, mem, cpu) utilization,
		// while the latter is in turn determined by the most recent and
		// maybe not so recent history of this target /selections/
		// by storage initiators
		//
		xvec := target.history[rclr.tdelay : rclr.tdelay+rclr.hisize]
		assert(len(xvec) == rclr.hisize)
		target.currlat = rclr.fnlatency(xvec)
	}
}

func score_I() {
OUTER:
	for _, initiator := range ivec {
		for _, target := range initiator.reptvec {
			if target == initiator.target {
				continue
			}
			// NOTE: > ... + fmin(epsilon, target.currlat/1000)
			if initiator.target.currlat > target.currlat {
				continue OUTER
			}
		}
		initiator.score++
	}
}

// reshuffle "jitters" to speed-up evolution
func rejitter_I(step int) {
	if !rclr.reusenoise || step%(rclr.ni-2) == 0 {
		for _, initiator := range ivec {
			initiator.genjitter = true
		}
		return
	}
	for _, initiator := range ivec {
		for l := 0; l < initiator.nn.lastidx; l++ {
			noisycube := initiator.gnoise[l]
			for jj := 0; jj < rclr.nperturb*2; jj++ {
				for r := 0; r < len(noisycube[jj]); r++ {
					row := noisycube[jj][r]
					a := row[0]
					copy(row[1:], row)
					row[len(row)-1] = a
					if len(row) > 3 {
						m := len(row) / 2
						for j := 1; j < m; j++ {
							a := row[m-j]
							row[m-j] = row[m+j]
							row[m+j] = a
						}
					}
				}
			}
		}
	}
}

// one mini-batch at a time (similar to gradientDescent)
func (initiator *Initiator) evolution() {
	b := initiator.nn.tunables.batchsize
	Xs, Ys := newMatrix(b, rclr.hisize), newVector(b)
	rewards := newVector(rclr.nperturb * 2)

	// iterate over NN layers, one layer's weights at a time
	num := initiator.nn.getHnum() + 1
	l := rand.Intn((initiator.id + 1) * 100)
	i := 0
	for {
		rclr.mstart.RLock()
		rclr.mstart.RUnlock()

		// fmt.Printf("--> %d\n", initiator.id)

		Xs[i] = cloneVector(initiator.distory)
		Ys[i] = initiator.target.currlat

		initiator.accumulateUpdates(Xs[i], Ys[i], (l % num), rewards, false)
		l++
		i++
		if i == b {
			initiator.evolve()
			if rclr.repeatevo {
				for ii := 0; ii < b; ii++ {
					fillVector(rewards, 0)
					initiator.accumulateUpdates(Xs[ii], Ys[ii], (l % num), rewards, true)
					l++
					initiator.evolve()
				}
			}
			i = 0
		}

		rclr.wg.Done()
		rclr.mend.RLock()
		rclr.mend.RUnlock()
	}
}

// averages the accumulated weight updates over a batchsize (compare with nn.Train())
func (initiator *Initiator) evolve() {
	nn := initiator.nn
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		divMatrixNum(layer.gradient, float64(nn.tunables.batchsize))
		addMatrixElem(layer.weights, layer.gradient)

		if rclr.usemomentum {
			mulMatrixNum(layer.pregradient, rclr.momentum)
			addMatrixElem(layer.weights, layer.pregradient)
			copyMatrix(layer.pregradient, layer.gradient)
		}
		zeroMatrix(layer.gradient)
	}
}

// "disturb" the weights (normally, at std 0.1) - to optimize the rewards
func (initiator *Initiator) accumulateUpdates(xvec []float64, y float64, l int, rewards []float64, repeating bool) {
	layer, layer_cpy := initiator.nn.layers[l], initiator.nn_cpy.layers[l]
	copyMatrix(layer_cpy.weights, layer.weights)

	// generate random normal jitter for the layer l weights
	// but only if the corresponding flag is on (otherwise we'd spend most of the local CPU time inside rand..)
	noisycube := initiator.gnoise[l]
	if !repeating && initiator.genjitter {
		for jj, j := 0, 0; j < rclr.nperturb; j++ {
			fillMatrixNormal(noisycube[jj], 0.0, 1.0, initiator.id, rclr.sparsity)
			mulMatrixNum(noisycube[jj], rclr.sigma)
			copyMatrix(noisycube[jj+1], noisycube[jj])
			mulMatrixNum(noisycube[jj+1], -1.0)
			jj += 2
		}
		initiator.genjitter = false
	}
	//
	// estimate the rewards for all 2*nperturb perturbations
	//
	initiator.nn.populateInput(xvec, true)
	for jj, j := 0, 0; j < rclr.nperturb; j++ {
		addMatrixElem(layer.weights, noisycube[jj])
		avec := initiator.nn.reForward()
		rewards[jj] = -math.Pow(avec[0]-y, 2)
		copyMatrix(layer.weights, layer_cpy.weights)
		//
		// and again, this time with a "negative" noise
		//
		addMatrixElem(layer.weights, noisycube[jj+1])
		avec = initiator.nn.reForward()
		rewards[jj+1] = -math.Pow(avec[0]-y, 2)
		copyMatrix(layer.weights, layer_cpy.weights)
		jj += 2
	}
	//
	// standardize the rewards and update the gradient
	//
	standardizeVectorZscore(rewards)
	for jj, j := 0, 0; j < rclr.nperturb; j++ {
		if rewards[jj] > rewards[jj+1]+rclr.rewardelta {
			if rewards[jj] > rclr.highreward {
				mulMatrixNum(noisycube[jj], rclr.hialpha*rewards[jj])
			} else {
				mulMatrixNum(noisycube[jj], initiator.nn.tunables.alpha*rewards[jj])
			}
			addMatrixElem(layer.gradient, noisycube[jj])
		} else if rewards[jj+1] > rewards[jj]+rclr.rewardelta {
			if rewards[jj+1] > rclr.highreward {
				mulMatrixNum(noisycube[jj+1], rclr.hialpha*rewards[jj+1])
			} else {
				mulMatrixNum(noisycube[jj+1], initiator.nn.tunables.alpha*rewards[jj+1])
			}
			addMatrixElem(layer.gradient, noisycube[jj+1])
		}
		jj += 2
	}
}

func (initiator *Initiator) gradientDescent() {
	b := initiator.nn.tunables.batchsize
	Xs, Ys := newMatrix(b, rclr.hisize), newMatrix(b, 1)
	ttp := &TTP{nn: initiator.nn, resultset: Ys, repeat: 3}
	i := 0
	for {
		rclr.mstart.RLock()
		rclr.mstart.RUnlock()

		// fmt.Printf("--> %d\n", initiator.id)

		Xs[i] = cloneVector(initiator.distory)

		Ys[i][0] = initiator.target.currlat
		i++
		if i == b {
			initiator.nn.Train(Xs, ttp)
			i = 0
		}

		rclr.wg.Done()
		rclr.mend.RLock()
		rclr.mend.RUnlock()
	}
}

func sumweights(nn *NeuNetwork) (wsum float64) {
	for l := 0; l < nn.lastidx; l++ {
		layer := nn.layers[l]
		next := layer.next
		for i := 0; i < layer.size; i++ {
			for j := 0; j < next.size; j++ {
				wsum += math.Abs(layer.weights[i][j])
			}
		}
	}
	wsum *= 10000
	return
}

func collab_I() {
	topscore, topidx := -1, -1
	for i, initiator := range ivec {
		if initiator.score >= topscore {
			topscore = initiator.score
			topidx = i
		}
	}
	// copy the winner
	first := ivec[topidx]

	// copy the winner
	for i, initiator := range ivec {
		if i == topidx {
			continue
		}
		w1 := float64(topscore) / float64(topscore+initiator.score)
		w2 := float64(initiator.score) / float64(topscore+initiator.score)
		if w1 > w2*rclr.scorehigh {
			fmt.Printf("%d(%.3f) --> %d(%.3f)\n", topidx, w1, i, w2)
			initiator.nn.reset()
			initiator.nn.copyNetwork(first.nn)
		}
	}
}

//==================================================================
//
// latency/reward functions:
// [[timed selection counts (aka history)]] ==> current latency
//
//==================================================================
func fn0_latency(h []float64) (lat float64) {
	lat = h[0]
	for i := 1; i < len(h); i++ {
		if h[i] == 0 && h[i-1] == 0 {
			break
		}
		if h[i] > 0 && h[i-1] > 0 {
			lat += lat * h[i] * 2
		} else {
			lat += lat * h[i]
		}
	}
	return
}

func fn1_latency(h []float64) (cv float64) {
	mean, std := meanStdVector(h)
	cv = std / (mean + DEFAULT_eps)
	return
}

// average
func fn3_latency(h []float64) (lat float64) {
	lat = meanVector(h)
	// To be precise, we'd need to convert average num of arrivals
	// (reads in this case) to the time units that measure latency.
	// But since this conversion is linear and fixed for the model,
	// we can simply return the historical mean
	//
	return
}

// constant processing time == leaky bucket:
// max(total-num-requests - num-requests-can-handle-in-so-many-steps), 0) + time-to-handle-the-last-one
// NOTE:
// non-linear factors such as burstiness and/or high utilization are disregarded
func fn4_latency(h []float64) (lat float64) {
	var reqs float64
	for i := len(h) - 1; i > 0; i-- {
		reqs += h[i] * float64(rclr.ni)
		if reqs > 0 {
			reqs -= rclr.svcrate
		}
		if reqs < 0 {
			reqs = 0
		}
	}
	reqs += h[0] * float64(rclr.ni)
	lat = reqs / rclr.svcrate
	return
}

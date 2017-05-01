package gorann

//
// multilayer NN is good to approximate about anything
//
import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// types
type Initiator struct {
	evo *Evolution
	// cluster
	reptvec []*Target
	// selected target, its estimated latency, and a copy of (delayed) history
	target  *Target
	minlat  float64
	distory []float64 // [[selectcnt / ni]]
	score   int
	id      int
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
// cluster single+config
//
type RCLR struct {
	wg        *sync.WaitGroup
	nextround *sync.Mutex
	cond      *sync.Cond
	step      int
	res1      int
	//
	svcrate                float64 // service rate
	scorehigh              float64 // scoring thresholds that control inter-initiator weights exchange
	ni, nt, nsteps, copies int     // numbers of initiators and targets, number of steps (epochs), number of replicas
	hisize                 int     // target (selection) history size
	coperiod               int     // colaboration = scoring period: number of steps (epochs)
	idelay, tdelay         int     // initiator and target "delay" as far as the history of the target /selections/
	//
	// FIXME: move the flags elsewhere
	repeatmini bool // "revolve" one more time on a given mini-batch
	useGD      bool // true - descend, false - evolve
	//
	fnlatency func(h []float64) float64
}

func init() {
	rclr = &RCLR{
		&sync.WaitGroup{},
		&sync.Mutex{},
		nil, 0, -1,
		1.1,            // service rate: num requests target can handle during one epoch step
		1.3,            // high scoring thresholds, to control inter-initiator weight exchange
		10, 10, 1E4, 3, // numbers of initiators and targets, num steps, replicas
		30,          //	target history size - selection counts that each target keeps back in time
		100,         // colaboration = scoring period: number of steps (epochs)
		0,           // (0, 1) => initiator and target see the same exact history
		1,           // (1, 0) => initiator's history is two epochs older than the target's
		false,       // re-evolve post weight-update, use the same mini-batch
		false,       // true - descend, false - evolve
		fn4_latency, //
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

	rclr.cond = sync.NewCond(rclr.nextround)
	rclr.nextround.Lock()
	rclr.wg.Add(rclr.ni)
	for _, initiator := range ivec {
		if rclr.useGD {
			go initiator.gradientDescent()
		} else {
			go initiator.revolution() // go run evolution
		}
	}

	rclr.wg.Wait()
	rclr.wg.Add(rclr.ni)

	for {
		if rclr.step > rclr.nsteps {
			break
		}
		compute_I() // I: select 3 rep-holding Ts, compute latencies based on (delayed) uvecs, select the T min
		compute_T() // T: given the selections (see prev step), compute "true" latencies and append to the uvecs
		score_I()   // I: +1 if selected the fastest target out of 3

		// synchronization block
		// fmt.Printf("========= %d =========\n", step)
		rclr.step++
		rclr.nextround.Unlock()
		rclr.cond.Broadcast()
		rclr.wg.Wait()
		rclr.nextround.Lock()
		rclr.wg.Add(rclr.ni)

		// inter-initiator weights exchange & printouts
		if rclr.step%rclr.coperiod == 0 {
			collab_I() // I: try to make use of the NN(s) with better results

			C, S := newVector(rclr.ni), newVector(rclr.ni)
			for i, initiator := range ivec {
				C[i] = math.Pow(initiator.minlat-initiator.target.currlat, 2)
				S[i] = float64(initiator.score) / float64(rclr.coperiod)

				// zero-out the scores for the next coperiod
				initiator.score = 0
			}
			fmt.Printf("%3d: %.3f\n", rclr.step/rclr.coperiod, S)
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
		input := NeuLayerConfig{size: rclr.hisize}
		hidden := NeuLayerConfig{"sigmoid", input.size * 2}
		output := NeuLayerConfig{"identity", 1}
		tu := &NeuTunables{gdalgname: ADAM, batchsize: 10, winit: XavierNewRand}
		etu := &EvoTunables{
			NeuTunables: *tu,
			sigma:       0.5,           // gaussian sigma aka std
			momentum:    0.01,          //
			hireward:    0.9,           // high(er) reward threshold and the corresponding learning rate
			hialpha:     0.01,          //
			rewd:        0.001,         //
			rewdup:      rclr.coperiod, // reward delta doubling period
			nperturb:    64,            // half the number of the NN weight perturbations aka jitters
			sparsity:    10,            // noise matrix sparsity (%)
			jinflate:    2}             // gaussian noise inflation ratio, to speed up evolution

		evo := NewEvolution(input, hidden, 2, output, etu, i)
		initiator := &Initiator{
			evo:     evo,
			reptvec: make([]*Target, rclr.copies),
			distory: newVector(rclr.hisize),
			id:      i}
		ivec[i] = initiator
	}
}

func compute_I() {
	for _, target := range tvec {
		target.selectcnt = 0
	}
	for _, initiator := range ivec {
		newrand := initiator.evo.newrand
		// 3 targets holding the replica that we want to read
		for i := 0; i < rclr.copies; i++ {
			initiator.reptvec[i] = tvec[newrand.Intn(rclr.nt)]
		}
		// estimate the respective latencies, and select
		initiator.target = nil
		for i := 0; i < rclr.copies; i++ {
			xvec := initiator.reptvec[i].history[rclr.idelay : rclr.idelay+rclr.hisize]
			assert(len(xvec) == rclr.hisize)
			avec := initiator.evo.Predict(xvec)
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

// run forever one mini-batch at a time (similar to gradientDescent)
func (initiator *Initiator) revolution() {
	newrand := initiator.evo.newrand
	newrand.Seed(int64(initiator.id))
	if initiator.evo.tunables.winit == XavierNewRand {
		initiator.evo.initXavier(newrand)
	}
	b := initiator.evo.tunables.batchsize
	Xs, Ys := newMatrix(b, rclr.hisize), newMatrix(b, 1)
	i, step := 0, 1

	rclr.wg.Done() // goroutine init done
	for {
		// wait for the siblings and for the main to do its part
		rclr.nextround.Lock()
		for rclr.step != step {
			rclr.cond.Wait()
		}
		rclr.nextround.Unlock()

		Xs[i] = cloneVector(initiator.distory)
		Ys[i][0] = initiator.target.currlat

		// Train(mini-batch) sequence:
		initiator.evo.TrainStep(Xs[i], Ys[i])
		i++
		if i == b {
			initiator.evo.fixWeights(b)
			if rclr.repeatmini {
				for ii := 0; ii < b; ii++ {
					initiator.evo.TrainStep(Xs[ii], Ys[ii])
				}
				initiator.evo.fixWeights(b)
			}
			i = 0
		}
		step++
		rclr.wg.Done() // goroutine step done
	}
}

func (initiator *Initiator) gradientDescent() {
	newrand := initiator.evo.newrand
	newrand.Seed(int64(initiator.id))
	initiator.evo.initXavier(newrand)

	b := initiator.evo.tunables.batchsize
	Xs, Ys := newMatrix(b, rclr.hisize), newMatrix(b, 1)
	ttp := &TTP{nn: &initiator.evo.NeuNetwork, resultset: Ys, repeat: 3}
	i := 0
	step := 1

	rclr.wg.Done() // goroutine init done

	for {
		// fmt.Printf("--> %d\n", initiator.id)
		rclr.nextround.Lock()
		for rclr.step != step {
			rclr.cond.Wait()
		}
		rclr.nextround.Unlock()

		Xs[i] = cloneVector(initiator.distory)

		Ys[i][0] = initiator.target.currlat
		i++
		if i == b {
			initiator.evo.Train(Xs, ttp)
			i = 0
		}
		step++
		rclr.wg.Done() // goroutine step done
	}
}

func collab_I() {
	topscore, topidx := -1, -1
	for i, initiator := range ivec {
		if initiator.score >= topscore {
			topscore = initiator.score
			topidx = i
		}
	}
	first := &ivec[topidx].evo.NeuNetwork
	for i, initiator := range ivec {
		if i == topidx {
			continue
		}
		w1 := float64(topscore) / float64(topscore+initiator.score)
		w2 := float64(initiator.score) / float64(topscore+initiator.score)
		if w1 > w2*rclr.scorehigh {
			fmt.Printf("%d(%.3f) --> %d(%.3f)\n", topidx, w1, i, w2)
			nn := &initiator.evo.NeuNetwork
			nn.reset()
			nn.copyNetwork(first)
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

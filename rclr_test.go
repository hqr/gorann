package gorann

//
// multilayer NN is good to approximate about anything
//
import (
	"fmt"
	"github.com/gonum/stat"
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
)

// types
type Initiator struct {
	evo *Evolution
	// cluster
	reptvec []*Target
	// selected target, its estimated latency, and a copy of (delayed) history
	target   *Target
	minlat   float64
	distory  []float64 // [[selectcnt / ni]]
	avgcost  float64
	scoreLat int
	scoreSel int
	id       int
}

type Target struct {
	history   []float64 // [[selectcnt / ni]]
	currlat   float64
	selectcnt int
	id        int
}

type ByLatency []*Initiator // implements sort.Interface

// static
var ivec, ivecsorted []*Initiator
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
	// cluster-wide tunables
	//
	svcrate                float64 // service rate
	scorehigh              float64 // scoring thresholds that control inter-initiator weights exchange
	ni, nt, nsteps, copies int     // numbers of initiators and targets, number of steps (epochs), number of replicas
	hisize                 int     // target (selection) history size
	coperiod               int     // collaboration = scoring period: number of steps (epochs)
	idelay, tdelay         int     // initiator and target "delay" as far as the history of the target /selections/
	//
	useGD bool // true - descend, false - evolve
	//
	fnlatency func(h []float64) float64
}

func init() {
	rclr = &RCLR{
		&sync.WaitGroup{},
		&sync.Mutex{},
		nil, 0, -1,
		//
		// cluster-wide tunables
		//
		1.5,            // service rate: num requests target can handle during one epoch step
		123456,         //1.3, // FIXME: disabled: high scoring thresholds that controls NN copying between initiators
		10, 10, 1E4, 3, // numbers of initiators and targets, num steps, replicas
		30,          //	target history size - selection counts that each target keeps back in time
		200,         // collaboration = scoring period: number of steps (epochs)
		1,           // initiator "sees" the history delayed by so many epochs
		0,           // target owns the presense
		false,       // true - descend, false - evolve
		fn5_latency, //
	}
	ivec = make([]*Initiator, rclr.ni)
	ivecsorted = make([]*Initiator, rclr.ni)
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
		compute_I() // I: select 3 rep-holding Ts, compute latencies, select the NN-estimated min
		compute_T() // T: given target selections, compute true latencies
		score_I()   // I: score based on the sorted order, by initiator latency

		// synchronize with the (I) workers
		rclr.step++
		rclr.nextround.Unlock()
		rclr.cond.Broadcast()
		rclr.wg.Wait() // ===================>>> revolution | gradientDescent
		rclr.nextround.Lock()
		rclr.wg.Add(rclr.ni)

		// inter-initiator collaboration
		if rclr.step%rclr.coperiod == 0 {
			collab_I() // make use of clustered NN(s) that produced /better/ results

			scLat, scSel, Cost := newVector(rclr.ni), newVector(rclr.ni), newVector(rclr.ni)
			for i, initiator := range ivec {
				scLat[i] = float64(initiator.scoreLat) / float64(rclr.coperiod*rclr.ni)
				scSel[i] = float64(initiator.scoreSel) / float64(rclr.coperiod)
				b := initiator.evo.tunables.batchsize
				Cost[i] = initiator.avgcost * float64(b) / float64(rclr.coperiod)
				initiator.avgcost = 0
				initiator.scoreLat, initiator.scoreSel = 0, 0
			}
			fmt.Printf("%3d: %.3f - score min latency\n", rclr.step/rclr.coperiod, scLat)
			fmt.Printf("%3d: %.3f - score target selection\n", rclr.step/rclr.coperiod, scSel)
			fmt.Printf("%3d: %.3f - cost last sample\n", rclr.step/rclr.coperiod, Cost)
			fmt.Printf("%3d: %.3f\n", rclr.step/rclr.coperiod, stat.Correlation(scLat, scSel, nil))
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
		// output := NeuLayerConfig{"identity", 1}
		output := NeuLayerConfig{"sigmoid", 1}
		tu := &NeuTunables{gdalgname: ADAM, batchsize: 10, winit: XavierNewRand}
		etu := &EvoTunables{
			NeuTunables: *tu,
			sigma:       0.01,          // normal-noise(0, sigma)
			momentum:    0.01,          //
			hireward:    10.0,          // diff (in num stds) that warrants a higher learning rate
			hialpha:     0.01,          //
			rewd:        0.001,         // FIXME: consider using reward / 1000
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
	xvec := newVector(rclr.hisize)
	for _, initiator := range ivec {
		newrand := initiator.evo.newrand
		// 3 targets holding the replica that we want to read
		for i := 0; i < rclr.copies; i++ {
			initiator.reptvec[i] = tvec[newrand.Intn(rclr.nt)]
		}
		// estimate the respective latencies, and select
		initiator.target = nil
		for i := 0; i < rclr.copies; i++ {
			// this target selection history, delayed and shifted one epoch back
			copy(xvec[1:], initiator.reptvec[i].history[rclr.idelay:rclr.idelay+rclr.hisize])

			// as far as the presense, let's assume I'll select this target
			// and will be the only one having it...
			xvec[0] += 1.0 / float64(rclr.ni)

			// use NN to predict
			avec := initiator.evo.Predict(initiator.distory)
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

//========
//
// sorting and scoring
//
//========
func (ivec ByLatency) Len() int           { return len(ivec) }
func (ivec ByLatency) Swap(i, j int)      { ivec[i], ivec[j] = ivec[j], ivec[i] }
func (ivec ByLatency) Less(i, j int) bool { return ivec[i].target.currlat < ivec[j].target.currlat }

// initiators compete between themselves
func score_I() {
	copy(ivecsorted, ivec)
	sort.Sort(ByLatency(ivecsorted))

	lat := ivecsorted[0].target.currlat
	addscore := rclr.ni
	ivecsorted[0].scoreLat += addscore
	for i := 1; i < rclr.ni; i++ {
		initiator := ivecsorted[i]
		if initiator.target.currlat <= lat+lat/100 {
			initiator.scoreLat += addscore
		} else {
			addscore--
			initiator.scoreLat += addscore
		}
	}

OUTER:
	for _, initiator := range ivec {
		for _, target := range initiator.reptvec {
			if target == initiator.target {
				continue
			}
			if math.Abs(initiator.target.currlat-target.currlat) < fmin(target.currlat/100, initiator.target.currlat/100) {
				continue OUTER
			}
		}
		initiator.scoreSel++
	}
}

func (initiator *Initiator) score() int {
	return initiator.scoreLat
	// return initiator.scoreSel
}

// FIXME: which score do I use: scoreLat or scoreSel or else? Which one will converge??
func collab_I() {
	topscore, topidx := -1, -1
	for i, initiator := range ivec {
		if initiator.score() >= topscore {
			topscore = initiator.score()
			topidx = i
		}
	}
	first := &ivec[topidx].evo.NeuNetwork
	for i, initiator := range ivec {
		if i == topidx {
			continue
		}
		w1 := float64(topscore) / float64(topscore+initiator.score())
		w2 := float64(initiator.score()) / float64(topscore+initiator.score())
		if w1 > w2*rclr.scorehigh {
			fmt.Printf("%d(%.3f) --> %d(%.3f)\n", topidx, w1, i, w2)
			nn := &initiator.evo.NeuNetwork
			nn.reset()
			nn.copyNetwork(first)
		}
	}
}

// streaming mode execution:
// run forever, one sample and one mini-batch at a time
// (compare to gradientDescent() alternative below)
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

		copyVector(Xs[i], initiator.distory)
		Ys[i][0] = initiator.target.currlat

		initiator.evo.TrainStep(Xs[i], Ys[i])
		i++
		if i == b {
			initiator.evo.fixGradients(b)
			initiator.evo.fixWeights(b)
			i = 0
			initiator.evo.reForward()
			initiator.avgcost += initiator.evo.costfunction(Ys[b-1])
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
	ttp := &TTP{nn: &initiator.evo.NeuNetwork, resultset: Ys}
	i, step := 0, 1

	rclr.wg.Done() // goroutine init done
	for {
		// fmt.Printf("--> %d\n", initiator.id)
		rclr.nextround.Lock()
		for rclr.step != step {
			rclr.cond.Wait()
		}
		rclr.nextround.Unlock()

		copyVector(Xs[i], initiator.distory)
		Ys[i][0] = initiator.target.currlat
		i++
		if i == b {
			initiator.evo.Train(Xs, ttp)
			i = 0
			// FIXME: average over coperiod
			initiator.evo.reForward()
			initiator.avgcost += initiator.evo.costfunction(Ys[b-1])
		}
		step++
		rclr.wg.Done() // goroutine step done
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
	for i := 2; i < len(h); i++ {
		if h[i] == 0 && h[i-1] == 0 && h[i-2] == 0 {
			break
		}
		if h[i] > 0 && h[i-1] > 0 && h[i-2] > 0 {
			lat += lat * h[i] * 4
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
	// But since this conversion is linear and fixed for the model...
	//
	return
}

// constant processing time, as in: leaky bucket
// max(total-num-requests - num-requests-can-handle-in-so-many-steps), 0) + time-to-handle-the-last-one
// non-linear factors such as burstiness and high utilization are disregarded
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
	lat = reqs / rclr.svcrate / 20 // normalize
	return
}

// high index of dispersion must cause a spike..
func fn5_latency(h []float64) (lat float64) {
	mean, std := meanStdVector(h[0:10])
	D := std * std / (mean + DEFAULT_eps)
	lat = fn4_latency(h)
	if D > 0.2 {
		lat *= (2 + D)
	}
	return
}

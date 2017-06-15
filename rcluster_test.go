package gorann

//
// reading cluster first, trains and then utilizes NNs to compete against
// otherwise enhanced clustered readers;
// the reading is initiator-to-target direct with no intermediaies, while
// the cluster is assumed to store 3 (three) copies of each data/meta chunk;
// comparison in the final phase is double-blind, with random assignment
// of the initiators to reading groups, as per configured group policies...
//
import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// types
type Initiator struct {
	nnint NeuNetworkInterface
	r     *NeuRunner
	// cluster
	reptvec []*Target
	// selected target, its estimated latency, and a copy of (delayed) history
	target            *Target
	lat, cumlat       float64
	distory           []float64 // [[selectcnt / ni]]
	xvectmp, smavec   []float64
	totalscore, score int
	id, reserved      int
}

type Target struct {
	currlat   float64
	history   []float64 // [[selectcnt]]
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
	p *NeuParallel
	//
	// cluster-wide tunables
	//
	svcrate                   float64 // service rate
	maxsteps                  int     // total number of computations (aka steps aka p.compute() calls)
	presteps                  int     // num pretrain steps
	period                    int     // scoring & logging period = number of steps/epochs
	ni, nt, batchsize, copies int     // numbers of initiators and targets, batchsize, number of replicas
	hisize                    int     // history size (the history keeps target's selection counts at past timestamps)
	dispsize                  int     //
	idelay, tdelay            int     // initiator and target delay as far as the history...
	sparse                    int     // weight matrix sparsity
	keepsma, wsma             int     // stats: keep so-many last n-point averages
	verbose                   bool    // verbose
	compete_random            bool    // compete against random-selection
}

func init() {
	rclr = &RCLR{
		nil,
		2.9,      // service rate: num requests target handles in one epoch
		int(1E5), // max steps (100 * period)
		1000,     // num pretrain steps (1 period)
		1000,     // period: scoring/logging
		20,       // num initiators
		20,       // num targets
		50,       // batchsize
		3,        // replicas
		100,      // history size (the history keeps target's selection counts at past timestamps)
		30,       // most recent of the history to compute dispersion
		2,        // initiators see the history delayed by so-many epochs
		0,        // target "delay" must be zero
		30,       // weight matrix sparsity (%)
		32,       // stats: num SMAs to keep and the width of the "window"
		50,       // stats: num SMAs to keep and the width of the "window"
		false,    // verbose
		false,    // compete against random-selection
	}
}

func Test_rcluster(t *testing.T) {
	rand.Seed(0)
	if cli.int1 != 0 {
		rclr.ni, rclr.nt = cli.int1, cli.int1
	}
	if cli.int2 != 0 {
		rclr.compete_random = cli.int2&1 > 0
		rclr.verbose = cli.int2&2 > 0
	}
	if cli.int3 != 0 {
		rclr.hisize = cli.int3
	}
	if cli.int4 != 0 {
		rclr.maxsteps = cli.int4
	}
	// construct
	rclr.p = NewNeuParallel(rclr.batchsize * 100)
	ivec = make([]*Initiator, rclr.ni)
	tvec = make([]*Target, rclr.nt)
	construct_T()
	construct_I()

	ttp := &TTP{sequential: true, num: rclr.batchsize, maxbackprops: rclr.maxsteps}
	for _, initiator := range ivec {
		r := rclr.p.attach(initiator.nnint, initiator.id, false, ttp)
		initiator.r = r
	}
	// log config and start all goroutines
	if true {
		s := fmt.Sprintf("config: %+v\n", rclr)
		i := strings.Index(s, "svc")
		rclr.p.logger.Print("{", s[i:])
	}
	rclr.p.start()

	var yvec = []float64{0}
	scores := newVector(rclr.ni)

	// Phase I. Training
	for keepgoing := true; keepgoing; {
		for _, target := range tvec {
			target.selectcnt = 0
		}
		// read = {select 3 rep-holding Ts, compute latencies, select the NN-estimated min}
		for _, initiator := range ivec {
			// consistent hashing would be happening here..
			for j := 0; j < rclr.copies; j++ {
				initiator.reptvec[j] = tvec[initiator.r.newrand.Intn(rclr.nt)]
			}
			initiator.selectTarget_NN()
		}

		// compute true latencies given actual target selections
		for _, target := range tvec {
			target.computeLatency()
		}

		// add training sample
		for _, initiator := range ivec {
			yvec[0] = initiator.target.currlat
			initiator.r.addSample(initiator.distory, yvec)
		}
		// feed-forward and back-propagate in parallel
		keepgoing = rclr.p.compute()

		// pre-training
		if rclr.p.step <= rclr.presteps {
			continue
		}

		score_I()

		// updated scores, trace
		if rclr.p.step%rclr.period == 0 && keepgoing {
			for i, initiator := range ivec {
				initiator.totalscore += initiator.score
				scores[i] = float64(initiator.totalscore) / float64(rclr.p.step)
				initiator.score = 0
			}
			tracescores_I(rclr.p.step/rclr.period, rclr.p.step*100/rclr.maxsteps, scores)
		}
	}
	rclr.p.stop() // stop all goroutines

	//
	// Phase II. Competition: NN against perfectly informed (PI)
	//
	itags := []string{"N1", "N2", "D1", "D2", "PI", "R"}
	if !rclr.compete_random {
		itags = itags[:len(itags)-1] // exclude random
	}
	ii, jj := firstsecond_I()
	rclr.p.logger.Print("*")
	rclr.p.logger.Print("* N1 = ", ii, ", N2 = ", jj)
	rclr.p.logger.Print("*")
	for _, initiator := range ivec {
		initiator.totalscore, initiator.score = 0, 0
	}
	assert(rclr.idelay > 1)
	for posteps := 1; posteps <= rclr.maxsteps; posteps++ {
		for _, target := range tvec {
			target.selectcnt = 0
		}
		//
		// all initiators: use their per-initiator policy to read one of the random copies
		//
		for i, initiator := range ivec {
			// consistent hashing would be happening here
			for j := 0; j < rclr.copies; j++ {
				initiator.reptvec[j] = tvec[initiator.r.newrand.Intn(rclr.nt)]
			}
			switch tag_I(i, ii, jj, itags) {
			case "N1": // the #1 neural network
				initiator.selectTarget_NN()
			case "N2": // the #2 NN
				initiator.selectTarget_NN()
			case "D2": // delay = 2
				initiator.selectTarget_PI(rclr.idelay)
			case "D1": // delay = 1
				initiator.selectTarget_PI(rclr.idelay - 1)
			case "PI": // zero delay = perfect information
				initiator.selectTarget_PI(0)
			case "R": // random selection
				initiator.selectTarget_R()
			default:
				assert(false)
			}
		}

		// compute true latencies given the actual target selections above
		for _, target := range tvec {
			target.computeLatency()
		}
		// compute scores and average latencies
		score_I()
		sma_I(posteps)
		if posteps%(10*rclr.period) == 0 {
			for i, initiator := range ivec {
				initiator.totalscore += initiator.score
				scores[i] = float64(initiator.totalscore) / float64(rclr.p.step)
				initiator.score = 0
			}
			k := posteps / (10 * rclr.period)
			tracescores_I(k, posteps*100/rclr.maxsteps, scores)
		}
	}
	//
	// Phase III. Printout: group-by initiator.tag
	//
	for _, it := range itags {
		for i, initiator := range ivec {
			if it != tag_I(i, ii, jj, itags) {
				continue
			}
			stag := fmt.Sprintf("%-2s", it)
			ssma := fmt.Sprintf("%.2f", initiator.smavec)
			rclr.p.logger.Printf("%-2d %s %s", i, stag, strings.Trim(ssma, "[]"))
		}
	}
	rclr.p.logger.Printf("\n       avg   std\n")
	for _, it := range itags {
		for i, initiator := range ivec {
			if it != tag_I(i, ii, jj, itags) {
				continue
			}
			mean, std := meanStdVector(initiator.smavec)
			stag := fmt.Sprintf("%-2s", it)
			rclr.p.logger.Printf("%-2d %s %.3f %.3f\n", i, stag, mean, std)
		}
	}
}

var lastpct int

func tracescores_I(iter, pct int, scores []float64) {
	if rclr.verbose {
		rclr.p.logger.Printf("%2d: %.3f\n\n", iter, scores)
		return
	}
	if lastpct != pct && pct%5 == 0 {
		fmt.Printf(" = %d%%", pct)
		lastpct = pct
		if pct == 100 {
			fmt.Printf("\n")
		}
	}
}

func tag_I(i, ii, jj int, itags []string) (tag string) {
	l := len(itags) - 2 // exclude N1 and N2
	switch {
	case i == ii:
		tag = "N1"
	case i == jj:
		tag = "N2"
	case i%l == 0:
		tag = fmt.Sprintf("D%d", rclr.idelay)
	case i%l == 1:
		tag = fmt.Sprintf("D%d", rclr.idelay-1)
	case i%l == 2:
		tag = "PI"
	default:
		tag = "R"
	}
	return
}

func construct_T() {
	for i, _ := range tvec {
		hvec := newVector(rclr.hisize + rclr.idelay + rclr.tdelay + 1)
		tvec[i] = &Target{history: hvec, id: i}
	}
}

func construct_I() {
	for i, _ := range ivec {
		input := NeuLayerConfig{size: rclr.hisize}
		// diversify
		hsize := input.size*2 + i*3
		hidden := NeuLayerConfig{"tanh", hsize}
		output := NeuLayerConfig{"sigmoid", 1}
		gdalgname := Rprop
		if i%2 == 0 {
			gdalgname = RMSprop
		}
		tu := &NeuTunables{gdalgname: gdalgname, batchsize: rclr.batchsize, winit: Xavier}
		nn := NewNeuNetwork(input, hidden, 2, output, tu)
		nn.sparsify(nil, rclr.sparse)
		nn.layers[1].config.actfname = "relu"
		const d = 10.0
		normalize := func(vec []float64) {
			vec[0] /= d
			assert(vec[0] < 1.0, fmt.Sprintf("normalize %f", vec[0]))
		}
		denormalize := func(vec []float64) { vec[0] *= d }
		nn.callbacks = &NeuCallbacks{nil, normalize, denormalize}

		initiator := &Initiator{
			nnint:   nn,
			reptvec: make([]*Target, rclr.copies),
			distory: newVector(rclr.hisize),
			xvectmp: newVector(rclr.hisize),
			smavec:  newVector(rclr.keepsma),
			id:      i + 1}
		ivec[i] = initiator
	}
}

// initiators compete between themselves
func score_I() {
OUTER:
	for _, initiator := range ivec {
		for _, target := range initiator.reptvec {
			if target == initiator.target {
				continue
			}
			if target.currlat < initiator.target.currlat {
				continue OUTER
			}
		}
		initiator.score++
	}
}

// simple moving average
func sma_I(iter int) {
	if iter%rclr.wsma != 0 {
		for _, initiator := range ivec {
			initiator.cumlat += initiator.target.currlat
		}
		return
	}
	for _, initiator := range ivec {
		copy(initiator.smavec, initiator.smavec[1:])
		initiator.smavec[rclr.keepsma-1] = initiator.cumlat / float64(rclr.wsma)
		initiator.cumlat = 0
	}
}

func firstsecond_I() (int, int) {
	ii, jj, first, second := 0, 1, ivec[0].totalscore, ivec[1].totalscore
	if first < second {
		ii, jj = jj, ii
		first, second = second, first
	}
	for i, initiator := range ivec {
		if initiator.totalscore > first {
			jj = ii
			second = first
			ii = i
			first = initiator.totalscore
		} else if initiator.totalscore > second {
			jj = i
			second = initiator.totalscore
		}
	}
	return ii, jj
}

/*
 * methods
 */

// best target selection using an already trained NN
// (is called by initiator)
func (initiator *Initiator) selectTarget_NN() {
	initiator.target = nil
	for i := 0; i < rclr.copies; i++ {
		tgt := initiator.reptvec[i]
		h := tgt.history[rclr.idelay : rclr.idelay+rclr.hisize]
		copy(initiator.xvectmp, h)

		divVectorNum(initiator.xvectmp, float64(rclr.ni)) // normalize

		avec := initiator.nnint.Predict(initiator.xvectmp)
		if initiator.target == nil || initiator.lat > avec[0] {
			initiator.lat = avec[0]
			initiator.target = tgt
			copy(initiator.distory, initiator.xvectmp)
		}
	}
	initiator.target.selectcnt++
}

// best target selection using the actual target-side latency-computing function
// note: delay = 0 corresponds to the perfect-information scenario
// (is called by initiator)
func (initiator *Initiator) selectTarget_PI(delay int) {
	initiator.target = nil
	for i := 0; i < rclr.copies; i++ {
		tgt := initiator.reptvec[i]
		h := tgt.history[delay : delay+rclr.hisize]
		currlat := tgt._latency(h)
		if initiator.target == nil || initiator.lat > currlat {
			initiator.lat = currlat
			initiator.target = tgt
		}
	}
	initiator.target.selectcnt++
}

// random selection
func (initiator *Initiator) selectTarget_R() {
	initiator.target = initiator.reptvec[0]
	initiator.target.selectcnt++
}

// target's latency is defined by its (disk, mem, cpu) utilization,
// while the latter is in turn determined by the most recent and
// maybe not so recent history of this target /selections/
// by storage initiators
func (target *Target) computeLatency() {
	// make place in history
	copy(target.history[1:], target.history)
	target.history[0] = float64(target.selectcnt)
	target.currlat = target._latency(target.history[rclr.tdelay : rclr.tdelay+rclr.hisize])
}

//
// constant processing time (finite queue with a leaky bucket "shaper")
// disregards non-linear factors: burstiness, high utilization, etc.
//
func (target *Target) _latency(h []float64) (lat float64) {
	var reqs float64
	for i := len(h) - 1; i > 0; i-- {
		reqs += h[i]
		if reqs > 0 {
			reqs -= rclr.svcrate
		}
		if reqs < 0 {
			reqs = 0
		}
	}
	reqs += h[0]
	lat = reqs / rclr.svcrate
	return
}

//
// --------- alternative impl ----------
// same as above + an attempt to reflect the burstiness
// as a high index of dispersion on the last portion of the history
//
func (target *Target) _latency_aux(h []float64) (lat float64) {
	mean, std := meanStdVector(h[0:rclr.dispsize])
	D := std * std / (mean + DEFAULT_eps)
	lat = target._latency(h)
	if D > 0.2 {
		lat *= (2 + D)
	}
	return
}

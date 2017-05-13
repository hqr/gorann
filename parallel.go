package gorann

import (
	// "fmt"
	"math/rand"
	"sync"
)

type NeuRunner struct {
	nnint   NeuNetworkInterface
	p       *NeuParallel
	newrand *rand.Rand
	Xb      [][]float64
	Yb      [][]float64
	avgcost float64
	id      int
	step    int
	i, j    int
	stopped bool
}

type NeuParallel struct {
	rumap map[int]*NeuRunner
	//
	wg        *sync.WaitGroup
	nextround *sync.Mutex
	cond      *sync.Cond
	step      int
}

func NewNeuParallel() *NeuParallel {
	p := &NeuParallel{
		make(map[int]*NeuRunner, 8),
		&sync.WaitGroup{},
		&sync.Mutex{},
		nil,
		0,
	}
	p.cond = sync.NewCond(p.nextround)
	return p
}

//
// NeuParallel methods
//
func (p *NeuParallel) attach(nnint NeuNetworkInterface, id int) *NeuRunner {
	b := nnint.getTunables().batchsize
	X := newMatrix(b, nnint.getIsize())
	Y := newMatrix(b, nnint.getCoutput().size)

	r := &NeuRunner{nnint: nnint, p: p, Xb: X, Yb: Y, id: id}
	r.newrand = rand.New(rand.NewSource(int64((id + 1) * 100)))
	r.newrand.Seed(int64(id))

	p.rumap[id] = r
	return r
}

func (p *NeuParallel) detach(id int) {
	delete(p.rumap, id)
}

func (p *NeuParallel) start() {
	p.nextround.Lock()
	p.wg.Add(len(p.rumap))

	for _, r := range p.rumap {
		go r.run()
	}
	p.wg.Wait()
	p.wg.Add(len(p.rumap))
	for _, r := range p.rumap {
		assert(r.step == 1) // in sync, step wise
	}
}

// graceful
func (p *NeuParallel) stop() {
	for _, r := range p.rumap {
		r.stopped = true
	}
}

// must be called periodically (from a caller's training loop)
func (p *NeuParallel) compute() {
	p.step++
	p.nextround.Unlock()
	p.cond.Broadcast()
	p.wg.Wait() // ===================>>> TrainStep by NeuRunner(s)
	p.nextround.Lock()
	p.wg.Add(len(p.rumap))
}

//
// NeuRunner methods
//
func (r *NeuRunner) addSample(xvec []float64, yvec []float64) {
	assert(r.i < len(r.Xb))
	copyVector(r.Xb[r.i], xvec)
	copyVector(r.Yb[r.i], yvec)
	r.i++
}

func (r *NeuRunner) run() {
	// init
	if r.nnint.getTunables().winit == XavierNewRand {
		r.nnint.initXavier(r.newrand)
	}
	b := r.nnint.getTunables().batchsize
	r.step = 1
	r.p.wg.Done()

	// work rounds until stopped
	for !r.stopped {
		// wait for the siblings and for the main to do its part
		r.p.nextround.Lock()
		for r.p.step != r.step {
			r.p.cond.Wait()
		}
		r.p.nextround.Unlock()

		// compute
		for r.j < r.i {
			TrainStep(r.nnint, r.Xb[r.j], r.Yb[r.j])
			r.j++
		}
		if r.i == b {
			r.nnint.fixGradients(b)
			r.nnint.fixWeights(b)
			r.i, r.j = 0, 0
			// optionally:
			r.nnint.reForward()
			r.avgcost += r.nnint.costfunction(r.Yb[b-1])
		}
		r.step++
		// signal the parallel parent
		r.p.wg.Done()
	}
}

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
	trwin   *TreamWin // ttp stream window
	avgcost float64
	id      int
	step    int
	trained int
	fixed   int
	stopped bool
}

type NeuParallel struct {
	rumap map[int]*NeuRunner
	//
	wg        *sync.WaitGroup
	nextround *sync.Mutex
	cond      *sync.Cond
	step      int
	size      int // ttp stream window size
}

func NewNeuParallel(size int) *NeuParallel {
	p := &NeuParallel{
		make(map[int]*NeuRunner, 8),
		&sync.WaitGroup{},
		&sync.Mutex{},
		nil,
		0,
		size,
	}
	p.cond = sync.NewCond(p.nextround)
	return p
}

//
// NeuParallel methods
//
func (p *NeuParallel) attach(nnint NeuNetworkInterface, id int) *NeuRunner {
	trwin := NewTreamWin(p.size, nnint)
	r := &NeuRunner{nnint: nnint, p: p, trwin: trwin, id: id}

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
	r.trwin.addSample(xvec, yvec)
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
		for r.trained < r.trwin.getSidx()+r.trwin.getSize() {
			r.nnint.TrainStep(r.trwin.getXvec(r.trained), r.trwin.getYvec(r.trained))
			r.trained++
		}
		if r.trained-r.fixed == b {
			r.nnint.fixGradients(b)
			r.nnint.fixWeights(b)
			r.fixed = r.trained
			// optionally:
			r.nnint.reForward()
			r.avgcost += r.nnint.costfunction(r.trwin.getYvec(r.trained - 1))
		}
		r.step++
		// signal the parallel parent
		r.p.wg.Done()
	}
}

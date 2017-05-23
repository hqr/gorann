package gorann

import (
	"log"
	"math/rand"
	"os"
	"sync"
)

type NeuRunner struct {
	nnint   NeuNetworkInterface
	p       *NeuParallel
	newrand *rand.Rand
	trwin   *TreamWin // ttp stream window
	ttp     *TTP
	id      int
	step    int
	trained int
	fixed   int
	stopped bool
	batch   bool
	//
	compute func()
}

type NeuParallel struct {
	rumap map[int]*NeuRunner
	//
	wg        *sync.WaitGroup
	nextround *sync.Mutex
	cond      *sync.Cond
	//
	logger *log.Logger
	//
	step int
	size int // ttp stream window size
}

func NewNeuParallel(size int) *NeuParallel {
	p := &NeuParallel{
		rumap:     make(map[int]*NeuRunner, 8),
		wg:        &sync.WaitGroup{},
		nextround: &sync.Mutex{},
		size:      size,
	}
	p.cond = sync.NewCond(p.nextround)
	p.logger = log.New(os.Stderr, "", 0) // log.LstdFlags)
	return p
}

//
// NeuParallel methods
//
func (p *NeuParallel) attach(nnint NeuNetworkInterface, id int, batch bool, ttp *TTP) *NeuRunner {
	assert(id > 0)
	r := NewNeuRunner(nnint, p, id, batch, ttp)
	p.rumap[id] = r
	return r
}

func (p *NeuParallel) detach(id int) {
	delete(p.rumap, id)
}

func (p *NeuParallel) get(id int) *NeuRunner {
	return p.rumap[id]
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

func NewNeuRunner(nnint NeuNetworkInterface, p *NeuParallel, id int, batch bool, ttp *TTP) *NeuRunner {
	trwin := NewTreamWin(p.size, nnint)
	r := &NeuRunner{nnint: nnint, p: p, trwin: trwin, id: id, batch: batch}

	r.newrand = rand.New(rand.NewSource(int64((id + 1) * 100)))
	r.newrand.Seed(int64(id))

	if batch {
		r.compute = r.computeBatch
	} else {
		r.compute = r.computeStream
	}
	r.ttp = &TTP{nn: nnint, runnerid: id, logger: p.logger} // NOTE: !sequential
	if ttp != nil {
		copyStruct(r.ttp, ttp)
		r.ttp.nn = nnint
		r.ttp.runnerid = id
		r.ttp.logger = p.logger
	}
	return r
}

func (r *NeuRunner) addSample(xvec []float64, yvec []float64) {
	r.trwin.addSample(xvec, yvec)
}

func (r *NeuRunner) run() {
	// init
	if r.nnint.getTunables().winit == XavierNewRand {
		r.nnint.initXavier(r.newrand)
	}
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

		// compute: stream | batch
		r.compute()

		r.step++
		// signal the parallel parent
		r.p.wg.Done()
	}
}

func (r *NeuRunner) computeStream() {
	b := r.nnint.getTunables().batchsize
	for r.trained < r.trwin.getSidx()+r.trwin.getSize() {
		r.nnint.TrainStep(r.trwin.getXvec(r.trained), r.trwin.getYvec(r.trained))
		r.trained++
	}
	if r.trained-r.fixed == b {
		r.nnint.fixGradients(b)
		r.nnint.fixWeights(b)
		r.fixed = r.trained
	}
}

func (r *NeuRunner) computeBatch() {
	assert(r.trained == r.fixed)

	cnv := r.ttp.Train(r.trwin)

	r.trained += r.trwin.getSize()
	r.fixed = r.trained

	if cnv > 0 {
		r.stopped = true
	}
}

func (r *NeuRunner) getCost() float64 {
	if r.batch {
		return r.ttp.avgcost
	}
	r.nnint.reForward()
	return r.nnint.costfunction(r.trwin.getYvec(r.trained - 1))
}

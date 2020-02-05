// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"testing"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

type Dtrevc3er interface {
	Dtrevc3(side lapack.EVSide, howmny lapack.EVHowMany, selected []bool, n int, t []float64, ldt int, vl []float64, ldvl int, vr []float64, ldvr int, mm int, work []float64, lwork int) int
}

func Dtrevc3Test(t *testing.T, impl Dtrevc3er) {
	rnd := rand.New(rand.NewSource(1))

	// Don't test with lapack.EVBoth because the handling is trivial in Dtrevc3
	// and it saves time.
	for _, side := range []lapack.EVSide{lapack.EVRight, lapack.EVLeft} {
		name := "Right"
		if side == lapack.EVLeft {
			name = "Left"
		}
		t.Run(name, func(t *testing.T) {
			runDtrevc3Test(t, impl, rnd, side)
		})
	}
}

func runDtrevc3Test(t *testing.T, impl Dtrevc3er, rnd *rand.Rand, side lapack.EVSide) {
	for _, n := range []int{0, 1, 2, 3, 4, 5, 6, 7, 10, 34} {
		for _, extra := range []int{0, 11} {
			for _, optwork := range []bool{true, false} {
				for cas := 0; cas < 10; cas++ {
					dtrevc3Test(t, impl, side, n, extra, optwork, rnd)
				}
			}
		}
	}
}

func dtrevc3Test(t *testing.T, impl Dtrevc3er, side lapack.EVSide, n, extra int, optwork bool, rnd *rand.Rand) {
	const tol = 1e-15

	right := side != lapack.EVLeft
	left := side != lapack.EVRight

	tmat, wr, wi := randomSchurCanonical(n, n+extra, rnd)
	tmatCopy := cloneGeneral(tmat)

	name := fmt.Sprintf("n=%d,extra=%d,optwk=%v", n, extra, optwork)

	// (1) Compute all eigenvectors.

	howmny := lapack.EVAll

	var vr, vl blas64.General
	if right {
		vr = nanGeneral(n, n, n+extra)
	}
	if left {
		vl = nanGeneral(n, n, n+extra)
	}

	var work []float64
	if optwork {
		work = []float64{0}
		impl.Dtrevc3(side, howmny, nil, n, tmat.Data, tmat.Stride,
			vl.Data, max(1, vl.Stride), vr.Data, max(1, vr.Stride), n, work, -1)
		work = make([]float64, int(work[0]))
	} else {
		work = make([]float64, max(1, 3*n))
	}

	mGot := impl.Dtrevc3(side, howmny, nil, n, tmat.Data, tmat.Stride,
		vl.Data, max(1, vl.Stride), vr.Data, max(1, vr.Stride), n, work, len(work))

	if !generalOutsideAllNaN(tmat) {
		t.Errorf("%v: out-of-range write to T", name)
	}
	if !generalOutsideAllNaN(vr) {
		t.Errorf("%v: out-of-range write to VR", name)
	}
	if !generalOutsideAllNaN(vl) {
		t.Errorf("%v: out-of-range write to VL", name)
	}

	mWant := n
	if mGot != mWant {
		t.Errorf("%v: unexpected value of m=%d, want %d", name, mGot, mWant)
	}

	if right {
		resid := residualRightEV(tmatCopy, vr, wr, wi)
		if resid > tol {
			t.Errorf("%v: unexpected right eigenvectors; residual=%v, want<=%v", name, resid, tol)
		}
		resid = residualEVNormalization(vr, wi)
		if resid > tol {
			t.Errorf("%v: unexpected normalization of right eigenvectors; residual=%v, want<=%v", name, resid, tol)
		}
	}
	if left {
		resid := residualLeftEV(tmatCopy, vl, wr, wi)
		if resid > tol {
			t.Errorf("%v: unexpected left eigenvectors; residual=%v, want<=%v", name, resid, tol)
		}
		resid = residualEVNormalization(vl, wi)
		if resid > tol {
			t.Errorf("%v: unexpected normalization of left eigenvectors; residual=%v, want<=%v", name, resid, tol)
		}
	}

	// (2) Compute selected eigenvectors and compare them to the full case.

	howmny = lapack.EVSelected

	// Follow DCHKHS and select last max(1,n/4) real, max(1,n/4) complex eigenvectors.
	selected := make([]bool, n)
	selectedWant := make([]bool, n)
	var nselr, nselc int
	for j := n - 1; j > 0; {
		if wi[j] == 0 {
			if nselr < max(1, n/4) {
				nselr++
				selected[j] = true
				selectedWant[j] = true
			}
			j--
		} else {
			if nselc < max(1, n/4) {
				nselc++
				selected[j] = true
				selected[j-1] = true
				selectedWant[j] = false
				selectedWant[j-1] = true
			}
			j -= 2
		}
	}
	mWant = nselr + 2*nselc

	if optwork {
		// Reallocate optimal work in case it depends on howmny and selected.
		work = []float64{0}
		impl.Dtrevc3(side, howmny, selected, n, tmat.Data, tmat.Stride,
			vl.Data, max(1, vl.Stride), vr.Data, max(1, vr.Stride), mWant, work, -1)
		work = make([]float64, int(work[0]))
	}

	copyGeneral(tmat, tmatCopy)

	var vrSel, vlSel blas64.General
	if right {
		vrSel = nanGeneral(n, mWant, n+extra)
	}
	if left {
		vlSel = nanGeneral(n, mWant, n+extra)
	}

	mGot = impl.Dtrevc3(side, howmny, selected, n, tmat.Data, tmat.Stride,
		vlSel.Data, max(1, vlSel.Stride), vrSel.Data, max(1, vrSel.Stride), mWant, work, len(work))

	if !generalOutsideAllNaN(tmat) {
		t.Errorf("%v: out-of-range write to T", name)
	}
	if !generalOutsideAllNaN(vrSel) {
		t.Errorf("%v: out-of-range write to selected VR", name)
	}
	if !generalOutsideAllNaN(vlSel) {
		t.Errorf("%v: out-of-range write to selected VL", name)
	}

	if mGot != mWant {
		t.Errorf("%v: unexpected value of selected m=%d, want %d", name, mGot, mWant)
	}

	for i := range selected {
		if selected[i] != selectedWant[i] {
			t.Errorf("%v: unexpected selected[%v]", name, i)
		}
	}

	var k int
	match := true
	if right {
	loopVR:
		for j := 0; j < n; j++ {
			if selected[j] && wi[j] == 0 {
				for i := 0; i < n; i++ {
					if vrSel.Data[i*vrSel.Stride+k] != vr.Data[i*vr.Stride+j] {
						match = false
						break loopVR
					}
				}
				k++
			} else if selected[j] && wi[j] != 0 {
				for i := 0; i < n; i++ {
					if vrSel.Data[i*vrSel.Stride+k] != vr.Data[i*vr.Stride+j] ||
						vrSel.Data[i*vrSel.Stride+k+1] != vr.Data[i*vr.Stride+j+1] {
						match = false
						break loopVR
					}
				}
				k += 2
			}
		}
	}
	if !match {
		t.Errorf("%v: unexpected selected VR", name)
	}

	match = true
	if left {
	loopVL:
		for j := 0; j < n; j++ {
			if selected[j] && wi[j] == 0 {
				for i := 0; i < n; i++ {
					if vlSel.Data[i*vlSel.Stride+k] != vl.Data[i*vl.Stride+j] {
						match = false
						break loopVL
					}
				}
				k++
			} else if selected[j] && wi[j] != 0 {
				for i := 0; i < n; i++ {
					if vlSel.Data[i*vlSel.Stride+k] != vl.Data[i*vl.Stride+j] ||
						vlSel.Data[i*vlSel.Stride+k+1] != vl.Data[i*vl.Stride+j+1] {
						match = false
						break loopVL
					}
				}
				k += 2
			}
		}
	}
	if !match {
		t.Errorf("%v: unexpected selected VL", name)
	}

	// (3) Compute all eigenvectors and multiply them into Q (the identity) and
	// compare the result to the full case.

	howmny = lapack.EVAllMulQ

	// var vrMul, vlMul blas64.General
	// if right {
	// 	vrMul = eye(n, n+extra)
	// }
	// if left {
	// 	vlMul = eye(n, n+extra)
	// }

}

// residualEVNormalization returns the maximum normalization error in E:
//  max |max-norm(E[:,j]) - 1|
func residualEVNormalization(e blas64.General, wi []float64) float64 {
	n := e.Rows
	if n == 0 {
		return 0
	}
	var (
		enrmin = math.Inf(1)
		enrmax float64
		ipair  int
	)
	for j := 0; j < n; j++ {
		if ipair == 0 && j < n-1 && wi[j] != 0 {
			ipair = 1
		}
		var nrm float64
		switch ipair {
		case 0:
			// Real eigenvector
			for i := 0; i < n; i++ {
				nrm = math.Max(nrm, math.Abs(e.Data[i*e.Stride+j]))
			}
			enrmin = math.Min(enrmin, nrm)
			enrmax = math.Max(enrmax, nrm)
		case 1:
			// Complex eigenvector
			for i := 0; i < n; i++ {
				nrm = math.Max(nrm, math.Abs(e.Data[i*e.Stride+j])+math.Abs(e.Data[i*e.Stride+j+1]))
			}
			enrmin = math.Min(enrmin, nrm)
			enrmax = math.Max(enrmax, nrm)
			ipair = 2
		case 2:
			ipair = 0
		}
	}
	return math.Max(math.Abs(enrmin-1), math.Abs(enrmin-1))
}

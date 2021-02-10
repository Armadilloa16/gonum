// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stat

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/floats"
)

func TestROC(t *testing.T) {
	const tol = 1e-14

	cases := []struct {
		c          []bool
		w          []float64
		wantTPR    []float64
		wantFPR    []float64
	}{
		// Test cases were informed by using sklearn metrics.roc_curve when
		// cutoffs is nil, but all test cases (including when cutoffs is not
		// nil) where calculated manually.
		// Some differences exist between unweighted ROCs from our function
		// and metrics.roc_curve which appears to use integer cutoffs in that
		// case. sklearn also appears to do some magic that trims leading zeros
		// sometimes.
		{ // 0
			c:          []bool{false, true, false, true, true, true},
			wantTPR:    []float64{0, 0.25, 0.5, 0.75, 0.75, 1, 1},
			wantFPR:    []float64{0, 0, 0, 0, 0.5, 0.5, 1},
		},
		{ // 1
			c:          []bool{false, true, false, true, true, true},
			w:          []float64{4, 1, 6, 3, 2, 2},
			wantTPR:    []float64{0, 0.25, 0.5, 0.875, 0.875, 1, 1},
			wantFPR:    []float64{0, 0, 0, 0, 0.6, 0.6, 1},
		},
		{ // 2
			c:          []bool{true, false, true, false},
			wantTPR:    []float64{0, 0, 0.5, 0.5, 1},
			wantFPR:    []float64{0, 0.5, 0.5, 1, 1},
		},
		{ // 3
			c:          []bool{false, false, true, true},
			wantTPR:    []float64{0, 0.5, 1, 1, 1},
			wantFPR:    []float64{0, 0, 0, 0.5, 1},
		},
		{ // 4
			c:          []bool{false, false},
			wantTPR:    []float64{math.NaN(), math.NaN(), math.NaN()},
			wantFPR:    []float64{0, 0.5, 1},
		},
		{ // 5
			c:          []bool{false},
			wantTPR:    []float64{math.NaN(), math.NaN()},
			wantFPR:    []float64{0, 1},
		},
		{ // 6
			c:          []bool{true},
			wantTPR:    []float64{0, 1},
			wantFPR:    []float64{math.NaN(), math.NaN()},
		},
		{ // 7
			c:          []bool{},
			wantTPR:    nil,
			wantFPR:    nil,
		},
	}
	for i, test := range cases {
		gotTPR, gotFPR := ROC(test.c, test.w)
		if !floats.Same(gotTPR, test.wantTPR) && !floats.EqualApprox(gotTPR, test.wantTPR, tol) {
			t.Errorf("%d: unexpected TPR got:%v want:%v", i, gotTPR, test.wantTPR)
		}
		if !floats.Same(gotFPR, test.wantFPR) && !floats.EqualApprox(gotFPR, test.wantFPR, tol) {
			t.Errorf("%d: unexpected FPR got:%v want:%v", i, gotFPR, test.wantFPR)
		}
	}
}

// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stat

// ROC returns paired false positive rate (FPR) and true positive rate
// (TPR) values corresponding to cutoff points on the receiver operator
// characteristic (ROC) curve obtained when y is treated as a binary
// classifier for classes with weights. The cutoff thresholds used to
// calculate the ROC are returned in thresh such that tpr[i] and fpr[i]
// are the true and false positive rates for y >= thresh[i].
//
// The input y and cutoffs must be sorted, and values in y must correspond
// to values in classes and weights. SortWeightedLabeled can be used to
// sort y together with classes and weights.
//
// For a given cutoff value, observations corresponding to entries in y
// greater than the cutoff value are classified as false, while those
// less than or equal to the cutoff value are classified as true. These
// assigned class labels are compared with the true values in the classes
// slice and used to calculate the FPR and TPR.
//
// If weights is nil, all weights are treated as 1.
//
// If cutoffs is nil or empty, all possible cutoffs are calculated,
// resulting in fpr and tpr having length one greater than the number of
// unique values in y. Otherwise fpr and tpr will be returned with the
// same length as cutoffs. floats.Span can be used to generate equally
// spaced cutoffs.
//
// More details about ROC curves are available at
// https://en.wikipedia.org/wiki/Receiver_operating_characteristic
func ROC(classes []bool, weights []float64) (tpr, fpr []float64) {
	if weights != nil && len(classes) != len(weights) {
		panic("stat: slice length mismatch")
	}
	if len(classes) == 0 {
		return nil, nil
	}

	tpr = make([]float64, len(classes) + 1)
	fpr = make([]float64, len(classes) + 1)
	var nPos, nNeg float64
	for i, u := range classes {
		tpr[i+1] = tpr[i]
		fpr[i+1] = fpr[i]

		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		if u {
			nPos += w
			tpr[i+1] += w
		} else {
			nNeg += w
			fpr[i+1] += w
		}
	}

	invNeg := 1 / nNeg
	invPos := 1 / nPos
	// Convert negative counts to TPR and FPR.
	// Bins beyond the maximum value in y are skipped
	// leaving these fpr and tpr elements as zero.
	for i := range tpr {
		// Prevent fused float operations by
		// making explicit float64 conversions.
		tpr[i] = 1 - float64(tpr[i]*invPos)
		fpr[i] = 1 - float64(fpr[i]*invNeg)
	}
	for i, j := 0, len(tpr)-1; i < j; i, j = i+1, j-1 {
		tpr[i], tpr[j] = tpr[j], tpr[i]
		fpr[i], fpr[j] = fpr[j], fpr[i]
	}

	return tpr, fpr
}

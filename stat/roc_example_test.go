// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stat_test

import (
	"fmt"

	"gonum.org/v1/gonum/integrate"
	"gonum.org/v1/gonum/stat"
)

func ExampleROC_weighted() {
	classes := []bool{false, true, false, true, true, true}
	weights := []float64{4, 1, 6, 3, 2, 2}

	tpr, fpr := stat.ROC(classes, weights)
	fmt.Printf("true  positive rate: %v\n", tpr)
	fmt.Printf("false positive rate: %v\n", fpr)

	// Output:
	// true  positive rate: [0 0.25 0.5 0.875 0.875 1 1]
	// false positive rate: [0 0 0 0 0.6 0.6 1]
}

func ExampleROC_unweighted() {
	classes := []bool{false, true, false, true, true, true}

	tpr, fpr := stat.ROC(classes, nil)
	fmt.Printf("true  positive rate: %v\n", tpr)
	fmt.Printf("false positive rate: %v\n", fpr)

	// Output:
	// true  positive rate: [0 0.25 0.5 0.75 0.75 1 1]
	// false positive rate: [0 0 0 0 0.5 0.5 1]
}

func ExampleROC_threshold() {
	y := []float64{0.1, 0.4, 0.35, 0.8}
	classes := []bool{false, false, true, true}
	stat.SortWeightedLabeled(y, classes, nil)

	tpr, fpr := stat.ROC(classes, nil)
	fmt.Printf("true  positive rate: %v\n", tpr)
	fmt.Printf("false positive rate: %v\n", fpr)

	// Output:
	// true  positive rate: [0 0.5 0.5 1 1]
	// false positive rate: [0 0 0.5 0.5 1]
}

func ExampleROC_unsorted() {
	y := []float64{8, 7.5, 6, 5, 3, 0}
	classes := []bool{true, true, true, false, true, false}
	weights := []float64{2, 2, 3, 6, 1, 4}

	stat.SortWeightedLabeled(y, classes, weights)

	tpr, fpr := stat.ROC(classes, weights)
	fmt.Printf("true  positive rate: %v\n", tpr)
	fmt.Printf("false positive rate: %v\n", fpr)

	// Output:
	// true  positive rate: [0 0.25 0.5 0.875 0.875 1 1]
	// false positive rate: [0 0 0 0 0.6 0.6 1]
}

func ExampleROC_aUC() {
	classes := []bool{true, false, true, false}

	tpr, fpr := stat.ROC(classes, nil)

	// Compute Area Under Curve.
	auc := integrate.Trapezoidal(fpr, tpr)
	fmt.Printf("true  positive rate: %v\n", tpr)
	fmt.Printf("false positive rate: %v\n", fpr)
	fmt.Printf("auc: %v\n", auc)

	// Output:
	// true  positive rate: [0 0 0.5 0.5 1]
	// false positive rate: [0 0.5 0.5 1 1]
	// auc: 0.25
}

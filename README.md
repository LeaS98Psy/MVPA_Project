# MVPA on Haxby Dataset (Practice / Tutorial Project)

## Purpose

This project is a **practice implementation** of multivariate pattern analysis (MVPA) on a public fMRI dataset. It demonstrates a standard decoding pipeline using a linear SVM, leave-one-run-out cross-validation, and a permutation test.

The code is intended for educational purposes. It does **not** produce novel scientific results.

## Dataset Source

The analysis uses the **Haxby et al. (2001)** fMRI dataset, available through Nilearn and originally published at:

Haxby, J. V., Gobbini, M. I., Furey, M. L., Ishai, A., Schouten, J. L., & Pietrini, P. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science*, 293(5539), 2425–2430.

The data is distributed via [OpenNeuro](https://openneuro.org/datasets/ds000105) and loaded automatically using `nilearn.datasets.fetch_haxby()`.

## Author
Lea Stupan, M.Sc. Cognitive Neuroscience, University of Münster

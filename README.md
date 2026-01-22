# CannyEdgeDetector3D

3D adaptation of the Canny edge detector for **volumetric data represented as a stack of 2D grayscale slices** (e.g., CT scans).  
Compared to running 2D Canny independently per slice, this implementation uses **3D-aware gradients**, **3D non-maximum suppression**, and **26-neighbour hysteresis** to produce cleaner, thinner contours across the volume.

## What it does

Given a sequence of consecutive slices (single-channel `cv::Mat`), the algorithm detects object boundaries in the **3D neighbourhood**:

1. **3D Gaussian smoothing** (separable 3D Gaussian as a product of 1D kernels).
2. **3D Sobel-Feldman gradients** along X/Y (in-plane) and Z (between slices).
3. **3D non-maximum suppression** along the gradient direction.
4. **Double thresholding** (low/high).
5. **Hysteresis / edge tracking** using **26-connected** neighbourhood in 3D.

### Slice-spacing issue and gradient interpolation

In CT-like data, the distance between slices is often larger than an in-slice pixel, and the true spacing (in pixels) may be unknown. This can make the Z-gradient component disproportionately large and degrade edge thinning.

To mitigate this, the implementation approximates neighbour gradients between slices via a **linear interpolation (“spline”) step** controlled by `sobel_coef`. Conceptually, for each slice it adjusts/approximates neighbour gradient maps to better match the unknown inter-slice spacing, and uses those approximations during non-maximum suppression.

## Build

### Requirements

- C++ compiler with C++17 support (or newer)
- CMake
- OpenCV (C++)

### Build steps

```bash
git clone https://github.com/VNVid/CannyEdgeDetector3D.git
cd CannyEdgeDetector3D/src
mkdir -p build && cd build
cmake .. && make

## Run (example)

```bash
./EdgeDetector
```

The example reads a set of slices, runs `Canny3D`, and (optionally) writes results to disk depending on how the demo is configured.

### Parameters

`void DetectEdges(std::vector<cv::Mat>& images,
                  int low_threshold=50,
                  int high_threshold=150,
                  std::string writing_dir="",
                  double sobel_coef=1e-5,
                  int blur_ksize=5);`

- `images`: ordered slice stack (`CV_8UC1` recommended).
- `low_threshold`, `high_threshold`: thresholds for the double-threshold step.
- `writing_dir`: directory to save outputs (if empty, saving may be disabled by the demo).
- `sobel_coef`: coefficient controlling neighbour-slice gradient approximation/interpolation.
- `blur_ksize`: Gaussian kernel size (odd).

## Main components

- `Canny3D` (`canny.h/.cpp`): full 3D Canny pipeline.
- `GaussianBlur3D` (`blur.h/.cpp`): 3D Gaussian smoothing on a slice stack.
- `SobelOperator` (`sobel.h/.cpp`): 3D Sobel gradients + gradient direction components.
  - Direction components are represented per axis with values in `{ -1, 0, 1 }` indicating
    direction along/against the axis or no component.

## Notes on evaluation (from the project report)

Quality was evaluated against “ideal” contours from the CT-OCR-2022 dataset using
Type I / Type II error probabilities based on false negatives / false positives normalized by contour size.
A comparison with OpenCV’s 2D Canny shows that results depend on thresholds and slice blocks,
and that the interpolation step visibly improves contour thinness and reduces spurious responses in many cases.

## References

- J. Canny, “A Computational Approach to Edge Detection”, 1986.
- H. C. Schau, “Statistical filter for image feature extraction”, 1980.
- CT-OCR-2022 dataset: https://zenodo.org/record/7181065
- OpenCV: https://opencv.org/

## Documentation

The codebase is documented with Doxygen-style comments (see headers in `src/`).

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

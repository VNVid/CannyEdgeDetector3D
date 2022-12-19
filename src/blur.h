#ifndef BLUR_H
#define BLUR_H

#include <opencv2/core/core_c.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 3-dimensional Gaussian filter
 *
 * @class GaussianBlur3D
 * Applies 3-dimensional Gaussian filter
 * of given size to the images
 */
class GaussianBlur3D {
 public:
  GaussianBlur3D(std::vector<cv::Mat>& images) : images_(images) {}

  /**
   * @brief Blurres the images
   *
   * @param ksize Size of filter, should be odd
   *
   * @return Blurred images
   */
  std::vector<cv::Mat> Blur(size_t ksize);

 private:
  std::vector<cv::Mat> images_;

  /**
   * @brief Computes 3-dimensional Gaussian filter of given size
   *
   * @param ksize Size of filter, should be odd
   *
   * @return 3-dimensional Gaussian filter
   */
  std::vector<cv::Mat> CreateFilter(size_t ksize);
};

#endif
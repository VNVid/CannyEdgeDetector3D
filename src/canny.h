#ifndef CANNY_H
#define CANNY_H

#include <blur.h>
#include <opencv2/core/core_c.h>
#include <sobel.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <vector>

/**
 * @brief 3-dimensional Canny operator
 *
 * @class Canny3D
 *
 */
class Canny3D {
 public:
  /**
   * @brief 3-dimensional Canny operator
   *
   * @param images Given images
   * @param low_threshold 	First threshold for the hysteresis procedure
   * @param high_threshold 	Second threshold for the hysteresis procedure
   * @param writing_dir Path to directory for saving results
   * @param sobel_coef Interpolation coefficient for SobelOperator @see
   * SobelOperator
   * @param blur_ksize Size of Gaussian filter, should be odd @see
   * GaussianBlur3D
   */
  void DetectEdges(std::vector<cv::Mat>& images, int low_threshold = 50,
                   int high_threshold = 150, std::string writing_dir = "",
                   double sobel_coef = 1e-5, int blur_ksize = 5);

 private:
  /**
   * @brief Suppresses non-maximum along gradient direction
   *
   * @param sop Sobel operator with counted gradients
   *
   * @return Images of Suppressed gradients
   *
   */
  std::vector<cv::Mat> NonMaximumSuppression(SobelOperator& sop);

  /**
   * @brief Applies double thresholding to suppressed gradients
   *
   * If gradient's magnitude is lower than low_threshold, it is not in the edge.
   * If magnitude is between thresholds, it becomes a candidate for edge.
   * If magnitude is higher than threshold, it is supposed to be in the edge.
   *
   * @param edge_images Images of Suppressed gradients
   * @param low_threshold 	First threshold for the hysteresis procedure
   * @param high_threshold 	Second threshold for the hysteresis procedure
   *
   *
   */
  void DoubleThresholding(std::vector<cv::Mat>& edge_images, int low_threshold,
                          int high_threshold);

  /**
   * @brief Finds final edges
   *
   * Neighbours of edge pixel which are candidates for edge become edge pixels
   * as well.
   *
   * @param edge_images Images of gradients after double thresholding
   *
   *
   */
  void EdgeTrackingByHysteresis(std::vector<cv::Mat>& edge_images);
};

#endif
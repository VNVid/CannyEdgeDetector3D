#ifndef SOBEL_H
#define SOBEL_H

#include <opencv2/core/core_c.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 3-dimensional Sobel-Feldman filters
 *
 * @struct SobelFeldmanFilters
 * Contains Sobel-Feldman filters for all directions
 */
struct SobelFeldmanFilters {
  SobelFeldmanFilters() {
    zdir.kernel1.convertTo(zdir.kernel1, CV_32SC1);
    zdir.kernel2.convertTo(zdir.kernel2, CV_32SC1);
    zdir.kernel3.convertTo(zdir.kernel3, CV_32SC1);
    xdir.kernel1.convertTo(xdir.kernel1, CV_32SC1);
    xdir.kernel2.convertTo(xdir.kernel2, CV_32SC1);
    xdir.kernel3.convertTo(xdir.kernel3, CV_32SC1);
    ydir.kernel1.convertTo(ydir.kernel1, CV_32SC1);
    ydir.kernel2.convertTo(ydir.kernel2, CV_32SC1);
    ydir.kernel3.convertTo(ydir.kernel3, CV_32SC1);
  }
  struct ZDirection {
    ZDirection() = default;
    const cv::Mat kernel1 = (cv::Mat_<int>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
    const cv::Mat kernel2 = (cv::Mat_<int>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const cv::Mat kernel3 =
        (cv::Mat_<int>(3, 3) << -1, -2, -1, -2, -4, -2, -1, -2, -1);
  };
  struct XDirection {
    // positive in the "right" direction
    XDirection() = default;
    const cv::Mat kernel1 =
        (cv::Mat_<int>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
    const cv::Mat kernel2 =
        (cv::Mat_<int>(3, 3) << 2, 0, -2, 4, 0, -4, 2, 0, -2);
    const cv::Mat kernel3 =
        (cv::Mat_<int>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
  };
  struct YDirection {
    // positive in the "down" direction
    YDirection() = default;
    const cv::Mat kernel1 =
        (cv::Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    const cv::Mat kernel2 =
        (cv::Mat_<int>(3, 3) << 2, 4, 2, 0, 0, 0, -2, -4, -2);
    const cv::Mat kernel3 =
        (cv::Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
  };

  ZDirection zdir;
  XDirection xdir;
  YDirection ydir;
};

/**
 * @brief 3-dimensional Sobel operator
 *
 * @class SobelOperator
 * Computes gradients and their directions along 3 axes,
 * also computes gradients for interpolation slices
 */
class SobelOperator {
 public:
  /**
   * @brief 3-dimensional Sobel operator
   *
   * @param images Given images
   * @param coef Interpolation coefficient
   */
  SobelOperator(std::vector<cv::Mat>& images, double coef = 1e-5);

  /**
   * @brief Getter for gradients
   *
   * Returns gradients if already computed,
   * otherwise calls Count() @see Count()
   *
   * @return Vector of images containing gradients
   */
  std::vector<cv::Mat> getGradient() {
    if (!counted_) Count();
    return gradient_;
  }
  /**
   * @brief Getter for gradients direction
   *
   * Returns gradients directions along columns if already computed,
   * otherwise calls @see Count()
   *
   * @return Vector of images containing directions
   */
  std::vector<cv::Mat> getGradDirectionX() {
    if (!counted_) Count();
    return grad_dir_x_;
  }
  /**
   * @brief Getter for gradients direction
   *
   * Returns gradients directions along rows if already computed,
   * otherwise calls @see Count()
   *
   * @return Vector of images containing directions
   */
  std::vector<cv::Mat> getGradDirectionY() {
    if (!counted_) Count();
    return grad_dir_y_;
  }
  /**
   * @brief Getter for gradients direction
   *
   * Returns gradients directions along images if already computed,
   * otherwise calls @see Count()
   *
   * @return Vector of images containing directions
   */
  std::vector<cv::Mat> getGradDirectionZ() {
    if (!counted_) Count();
    return grad_dir_z_;
  }
  /**
   * @brief Getter for interpolated gradients
   *
   * Returns gradients if already computed,
   * otherwise calls @see Count()
   *
   * @return Vector of images containing gradients
   */
  std::vector<std::pair<cv::Mat, cv::Mat>> getNeighbourGrads() {
    if (!counted_) Count();
    return interpolated_gradient_;
  }

 private:
  // flag: true - if gradients and directions are counted
  bool counted_ = false;

  std::vector<cv::Mat> images_;
  std::vector<std::vector<cv::Mat>> interpolated_images_;  // of size 4
  std::vector<cv::Mat> gradient_;
  std::vector<std::pair<cv::Mat, cv::Mat>> interpolated_gradient_;
  // gradient direction
  std::vector<cv::Mat> grad_dir_x_;  // between columns
  std::vector<cv::Mat> grad_dir_y_;  // between rows
  std::vector<cv::Mat> grad_dir_z_;  // between images

  /**
   * @brief 3-dimensional Sobel operator
   *
   * All main computations
   */
  void Count();
};

#endif
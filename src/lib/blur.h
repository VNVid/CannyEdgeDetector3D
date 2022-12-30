#ifndef BLUR_H
#define BLUR_H

#include <opencv2/core/core_c.h>

#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#ifndef M_PI
namespace {
const double M_PI = std::acos(-1.0);
}
#endif

/**
 * @brief Трехмерный фильтр Гаусса
 *
 * @class GaussianBlur3D
 * Применяет трехмерный фильтр Гаусса заданного размера
 * к изображениям
 */
class GaussianBlur3D {
 public:
  GaussianBlur3D(std::vector<cv::Mat>& images) : images_(images) {}

  /**
   * @brief Размывает изображения
   *
   * @param ksize Размер фильтра, должен быть нечетным
   *
   * @return Массив размытых изображений
   */
  std::vector<cv::Mat> Blur(size_t ksize);

 private:
  std::vector<cv::Mat> images_;

  /**
   * @brief Вычисляет трехмерный фильтр Гаусса заданного размера
   *
   * @param ksize Размер фильтра, должен быть нечетным
   *
   * @return трехмерный фильтр Гаусса
   */
  std::vector<cv::Mat> CreateFilter(size_t ksize);
};

#endif
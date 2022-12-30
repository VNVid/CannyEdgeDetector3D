#ifndef SOBEL_H
#define SOBEL_H

#include <opencv2/core/core_c.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Трехмерные фильтры Собеля-Фельдмана
 *
 * @struct SobelFeldmanFilters
 * Содержит фильтры для всех направлений
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
 * @brief Трехмерный оператор Собеля
 *
 * @class SobelOperator
 * Считает градиенты и их направление вдоль 3х осей,
 * также считает градиенты для приближенных соседних срезов
 *
 */
class SobelOperator {
 public:
  /**
   * @brief Трехмерный оператор Собеля
   *
   * @param images Обрабатываемые изображения
   * @param coef Коэффициент приближения соседних срезов
   */
  SobelOperator(std::vector<cv::Mat>& images, double coef = 1e-5);

  /**
   * @brief Геттер для градиентов
   *
   * Если градиенты еще не посчитаны, вызывает метод Count() @see Count().
   * Возвращает значение градиентов.
   *
   * @return Массив карт градиентов
   */
  std::vector<cv::Mat> getGradient() {
    if (!counted_) Count();
    return gradient_;
  }
  /**
   * @brief Геттер для направлений градиентов
   *
   * Если градиенты еще не посчитаны, вызывает метод Count() @see Count().
   * Возвращает составляющую направлений градиентов вдоль столбцов.
   * Возможные значения: 0 - нет составляющей вдоль выбранной оси,
   * 1 - есть составляющая вдоль выбранной оси,
   * -1 - есть составляющая против выбранной оси
   *
   * @return Массив изображений со значениями направления
   */
  std::vector<cv::Mat> getGradDirectionX() {
    if (!counted_) Count();
    return grad_dir_x_;
  }
  /**
   * @brief Геттер для направлений градиентов
   *
   * Если градиенты еще не посчитаны, вызывает метод Count() @see Count().
   * Возвращает составляющую направлений градиентов вдоль строк.
   * Возможные значения: 0 - нет составляющей вдоль выбранной оси,
   * 1 - есть составляющая вдоль выбранной оси,
   * -1 - есть составляющая против выбранной оси
   *
   * @return Массив изображений со значениями направления
   */
  std::vector<cv::Mat> getGradDirectionY() {
    if (!counted_) Count();
    return grad_dir_y_;
  }
  /**
   * @brief Геттер для направлений градиентов
   *
   * Если градиенты еще не посчитаны, вызывает метод Count() @see Count().
   * Возвращает составляющую направлений градиентов вдоль оси, перпендикулярной
   * плоскости картинок. Возможные значения: 0 - нет составляющей вдоль
   * выбранной оси, 1 - есть составляющая вдоль выбранной оси, -1 - есть
   * составляющая против выбранной оси
   *
   * @return Массив изображений со значениями направления
   */
  std::vector<cv::Mat> getGradDirectionZ() {
    if (!counted_) Count();
    return grad_dir_z_;
  }
  /**
   * @brief Геттер для градиентов риближенных соседних срезов
   *
   * Если градиенты еще не посчитаны, вызывает метод Count() @see Count().
   * Возвращает значение градиентов.
   *
   * @return Массив карт градиентов
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
   * @brief Трехмерный оператор Собеля
   *
   * Выполняет все рассчеты
   */
  void Count();
};

#endif
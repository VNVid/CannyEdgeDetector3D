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
 * @brief Трехмерный оператор Кэнни
 *
 * @class Canny3D
 *
 */
class Canny3D {
 public:
  /**
   * @brief Трехмерный оператор Кэнни
   *
   * @param images Изображения, на которых нужно найти границы
   * @param low_threshold 	Нижний порог фильтрации
   * @param high_threshold 	Верхний порог фильтрации
   * @param sobel_coef Коэффициент приближения соседних срезов для оператора
   * Собеля @see SobelOperator
   * @param blur_ksize Размер фильтра Гаусса, должен быть нечетным @see
   * GaussianBlur3D
   */
  std::vector<cv::Mat> DetectEdges(std::vector<cv::Mat>& images,
                                   int low_threshold = 50,
                                   int high_threshold = 150,
                                   double sobel_coef = 1e-5,
                                   int blur_ksize = 5);

 private:
  /**
   * @brief Подавление немаксимумов вдоль направления градиента
   *
   * @param sop Оператор Собеля с посчитанными градиентами
   *
   * @return Массив карт градиентов после выполнения процедуры
   *
   */
  std::vector<cv::Mat> NonMaximumSuppression(SobelOperator& sop);

  /**
   * @brief Двойная пороговая фильтрация
   *
   * Если значение градиента пикселя меньше нижнего порога, то пиксель не
   * принадлежит границе. Если значение градиента пикселя больше верхнего
   * порога, то пиксель является граничным. Если значение между порогами, то
   * пиксель помечается, как кандидат в граничные пиксели.
   *
   * @param edge_images Карты градиентов после подавления немаксимумов
   * @param low_threshold 	Нижний порог фильтрации
   * @param high_threshold 	Верхний порог фильтрации
   *
   *
   */
  void DoubleThresholding(std::vector<cv::Mat>& edge_images, int low_threshold,
                          int high_threshold);

  /**
   * @brief Определяет границы
   *
   * Пиксели, которые были помечены как кандидаты в граничные и которые являются
   * соседними для граничных пикселей, также добавляются к границе
   * @param edge_images Массив карт градиентов после двойной пороговой
   * фильтрации
   *
   *
   */
  void EdgeTrackingByHysteresis(std::vector<cv::Mat>& edge_images);
};

#endif
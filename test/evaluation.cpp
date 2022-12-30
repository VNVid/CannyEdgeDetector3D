#include <cassert>
#include <cstdint>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Counts probability of type I and type II errors
std::pair<double, double> CountErrors(cv::Mat result, cv::Mat ideal) {
  assert(result.cols == ideal.cols && result.rows == ideal.rows);

  cv::Mat ideal_8u;
  cv::Mat result_8u;
  ideal.convertTo(ideal_8u, CV_8UC1);
  result.convertTo(result_8u, CV_8UC1);

  int false_negative = 0;
  int false_positive = 0;
  int ideal_edge_num = 0;
  int result_edge_num = 0;
  for (size_t i = 0; i < ideal_8u.rows; i++) {
    for (size_t j = 0; j < ideal_8u.cols; j++) {
      if (ideal_8u.at<uint8_t>(i, j) > 100 &&
          result_8u.at<uint8_t>(i, j) < 200) {
        // is an edge pixel but wasn't detected
        ++false_negative;
      }
      if (ideal_8u.at<uint8_t>(i, j) < 100 &&
          result_8u.at<uint8_t>(i, j) > 200) {
        // is not an edge pixel but is detected
        ++false_positive;
      }
      if (ideal_8u.at<uint8_t>(i, j) > 100) {
        ++ideal_edge_num;
      }
      if (result_8u.at<uint8_t>(i, j) > 200) {
        ++result_edge_num;
      }
    }
  }

  int edge_num = std::max(result_edge_num, ideal_edge_num);

  return std::make_pair((double)false_negative / edge_num,
                        (double)false_positive / edge_num);
}

int main() {
  /*std::cout << "Enter path to the result image\n";
  std::string path_result;
  std::cin >> path_result;
  std::cout << "Enter path to image with ideal edges\n";
  std::string path_ideal;
  std::cin >> path_ideal;

  cv::Mat result = cv::imread(path_result, cv::IMREAD_UNCHANGED);
  cv::Mat ideal = cv::imread(path_ideal, cv::IMREAD_GRAYSCALE);
  std::pair<double, double> errors = CountErrors(result, ideal);

  std::cout << errors.first << " " << errors.second << std::endl;*/

  std::vector<double> first_err;
  std::vector<double> second_err;
  for (size_t i = 3; i < 10; i++) {
    cv::Mat result = cv::imread("images/110" + std::to_string(i) + ".png",
                                cv::IMREAD_UNCHANGED);
    cv::Mat edges;
    cv::Canny(result, edges, 40, 160, 3, false);
    cv::Mat ideal =
        cv::imread("images/SPTV110" + std::to_string(i) + ".tif.seg.png",
                   cv::IMREAD_GRAYSCALE);
    std::pair<double, double> errors = CountErrors(edges, ideal);
    first_err.push_back(errors.first);
    second_err.push_back(errors.second);
  }
  for (size_t i = 10; i < 13; i++) {
    cv::Mat result = cv::imread("images/11" + std::to_string(i) + ".png",
                                cv::IMREAD_UNCHANGED);
    cv::Mat edges;
    cv::Canny(result, edges, 40, 160, 3, false);
    cv::Mat ideal =
        cv::imread("images/SPTV11" + std::to_string(i) + ".tif.seg.png",
                   cv::IMREAD_GRAYSCALE);
    std::pair<double, double> errors = CountErrors(edges, ideal);
    first_err.push_back(errors.first);
    second_err.push_back(errors.second);
  }

  for (double a : first_err) {
    std::cout << a << ", ";
  }
  std::cout << std::endl;
  for (double a : second_err) {
    std::cout << a << ", ";
  }
}
#include <canny.h>

#include <iostream>
#include <string>
#include <vector>

int main() {
  // reading preproccesssed images
  std::cout << "Reading images" << std::endl;
  std::vector<cv::Mat> images;
  for (size_t i = 1100; i < 1116; i++) {
    cv::Mat img = cv::imread("../slices/" + std::to_string(i) + ".png",
                             cv::IMREAD_UNCHANGED);
    images.push_back(img);
  }
  std::cout << "Images read" << std::endl;

  Canny3D canny;
  std::vector<cv::Mat> edges = canny.DetectEdges(images, 40, 180, 1);

  for (size_t i = 0; i < edges.size(); i++) {
    cv::imwrite(std::to_string(i) + ".png", edges[i]);
  }
}

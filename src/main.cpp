#include <canny.h>

#include <iostream>
#include <string>
#include <vector>

int main() {
  // reading preproccesssed images
  std::vector<cv::Mat> images;
  for (size_t i = 1100; i < 1116; i++) {
    cv::Mat img = cv::imread("../slices/" + std::to_string(i) + ".png",
                             cv::IMREAD_UNCHANGED);
    images.push_back(img);
  }
  std::cout << "Images read" << std::endl;

  Canny3D canny;
  canny.DetectEdges(images, 40, 180, "TH_40_180_SC_1(nointerpol)/", 1);
}

#include <blur.h>

std::vector<cv::Mat> GaussianBlur3D::Blur(size_t ksize) {
  // padding with empty images
  std::vector<cv::Mat> image_tmp;
  std::vector<cv::Mat> blurred;
  for (size_t i = 0; i < ksize / 2; i++) {
    image_tmp.push_back(cv::Mat::zeros(images_[0].size(), CV_64FC1));
  }
  for (cv::Mat img : images_) {
    cv::Mat mat;
    img.convertTo(mat, CV_64FC1);
    image_tmp.push_back(mat);
    blurred.push_back(mat);
  }
  for (size_t i = 0; i < ksize / 2; i++) {
    image_tmp.push_back(cv::Mat::zeros(images_[0].size(), CV_64FC1));
  }

  std::vector<cv::Mat> filter = CreateFilter(ksize);
  // applying Gaussian filter
  for (int img_i = ksize / 2; img_i < image_tmp.size() - ksize / 2; img_i++) {
    for (int i = ksize / 2; i < image_tmp[img_i].rows - ksize / 2; i++) {
      for (int j = ksize / 2; j < image_tmp[img_i].cols - ksize / 2; j++) {
        double value = 0;
        for (int kernel = 0; kernel < ksize; kernel++) {
          cv::Mat part = image_tmp[img_i - ksize / 2 + kernel](
              cv::Rect(j - ksize / 2, i - ksize / 2, ksize, ksize));

          value += cv::sum(part.mul(filter[kernel])).val[0];
        }
        blurred[img_i - ksize / 2].at<double>(i, j) = value;
      }
    }
  }

  std::vector<cv::Mat> result;
  for (cv::Mat img : blurred) {
    cv::Mat mat;
    img.convertTo(mat, CV_32SC1);
    result.push_back(mat);
  }
  return result;
}

std::vector<cv::Mat> GaussianBlur3D::CreateFilter(size_t ksize) {
  // compute sigma from kernel size like in OpenCV
  double sigma = 0.3 * (((double)ksize - 1) * 0.5 - 1) + 0.8;

  // creating filter
  double sum = 0;
  std::vector<cv::Mat> filter;
  for (size_t i = 0; i < ksize; i++) {
    filter.push_back(cv::Mat::zeros(ksize, ksize, CV_64FC1));
  }
  int k = ksize / 2 + 1;
  for (int x = 0; x < ksize; x++) {
    for (int y = 0; y < ksize; y++) {
      for (int z = 0; z < ksize; z++) {
        double exp_pow =
            -((x - k) * (x - k) + (y - k) * (y - k) + (z - k) * (z - k)) /
            (2 * sigma * sigma);
        filter[z].at<double>(x, y) = exp(exp_pow) / (2 * sigma * sigma * M_PI);
        sum += filter[z].at<double>(x, y);
      }
    }
  }

  // normalizing filter, so that sum of elements is 1
  for (size_t x = 0; x < ksize; x++) {
    for (size_t y = 0; y < ksize; y++) {
      for (size_t z = 0; z < ksize; z++) {
        filter[z].at<double>(x, y) /= sum;
      }
    }
  }

  return filter;
}
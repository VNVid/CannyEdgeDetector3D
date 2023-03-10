#include <canny.h>

std::vector<cv::Mat> Canny3D::DetectEdges(std::vector<cv::Mat>& images,
                                          int low_threshold, int high_threshold,
                                          double sobel_coef, int blur_ksize) {
  std::cout << "Applying Gaussian filter" << std::endl;
  // Gaussian filter
  std::vector<cv::Mat> blurred_images = GaussianBlur3D(images).Blur(blur_ksize);

  std::cout << "Counting gradients" << std::endl;
  SobelOperator sop(blurred_images, sobel_coef);

  std::cout << "Non-maximum suppression stage" << std::endl;
  std::vector<cv::Mat> edge_images = NonMaximumSuppression(sop);

  std::cout << "Double thresholding stage" << std::endl;
  DoubleThresholding(edge_images, low_threshold, high_threshold);

  std::cout << "Detecting edges" << std::endl;
  EdgeTrackingByHysteresis(edge_images);

  std::cout << "End of detection" << std::endl;
  return edge_images;
}

std::vector<cv::Mat> Canny3D::NonMaximumSuppression(SobelOperator& sop) {
  std::vector<cv::Mat> grads = sop.getGradient();
  std::vector<std::pair<cv::Mat, cv::Mat>> neighb_grads =
      sop.getNeighbourGrads();
  std::vector<cv::Mat> suppressed_grads = sop.getGradient();
  std::vector<cv::Mat> xdir = sop.getGradDirectionX();
  std::vector<cv::Mat> ydir = sop.getGradDirectionY();
  std::vector<cv::Mat> zdir = sop.getGradDirectionZ();

  for (size_t img_i = 1; img_i < grads.size() - 1; img_i++) {
    for (size_t i = 1; i < grads[img_i].rows - 1; i++) {
      for (size_t j = 1; j < grads[img_i].cols - 1; j++) {
        int dx = xdir[img_i].at<int32_t>(i, j);
        int dy = ydir[img_i].at<int32_t>(i, j);
        int dz = zdir[img_i].at<int32_t>(i, j);
        if (dz == 1) {
          dx = -dx;
          dy = -dy;
        }
        if (grads[img_i].at<int32_t>(i, j) <
                neighb_grads[img_i].first.at<int32_t>(i + dy, j + dx) ||
            grads[img_i].at<int32_t>(i, j) <
                neighb_grads[img_i].second.at<int32_t>(i - dy, j - dx)) {
          suppressed_grads[img_i].at<int32_t>(i, j) = 0;
        }
      }
    }

    cv::normalize(suppressed_grads[img_i], suppressed_grads[img_i], 0, 255,
                  cv::NORM_MINMAX);
  }

  // changing type to uint8_t
  std::vector<cv::Mat> result;
  for (auto grad : suppressed_grads) {
    cv::Mat img;
    grad.convertTo(img, CV_8U);
    result.push_back(img);
  }

  return result;
}

void Canny3D::DoubleThresholding(std::vector<cv::Mat>& edge_images,
                                 int low_threshold, int high_threshold) {
  for (size_t img_i = 1; img_i < edge_images.size() - 1; img_i++) {
    for (size_t i = 0; i < edge_images[img_i].rows; i++) {
      for (size_t j = 0; j < edge_images[img_i].cols; j++) {
        int value = edge_images[img_i].at<uint8_t>(i, j);

        if (value > high_threshold) {
          edge_images[img_i].at<uint8_t>(i, j) = 255;
        } else if (value < low_threshold) {
          edge_images[img_i].at<uint8_t>(i, j) = 0;
        } else {
          edge_images[img_i].at<uint8_t>(i, j) = 127;
        }
      }
    }
  }
}

void Canny3D::EdgeTrackingByHysteresis(std::vector<cv::Mat>& edge_images) {
  std::queue<int> candidate_row;
  std::queue<int> candidate_col;
  std::queue<int> candidate_img;
  for (size_t img_i = 1; img_i < edge_images.size() - 1; img_i++) {
    for (size_t row_i = 0; row_i < edge_images[img_i].rows; row_i++) {
      for (size_t col_j = 0; col_j < edge_images[img_i].cols; col_j++) {
        if (edge_images[img_i].at<uint8_t>(row_i, col_j) != 255) {
          continue;
        }

        candidate_row.push(row_i);
        candidate_col.push(col_j);
        candidate_img.push(img_i);
        while (!candidate_col.empty()) {
          int row = candidate_row.front();
          int col = candidate_col.front();
          int pic = candidate_img.front();
          candidate_col.pop();
          candidate_row.pop();
          candidate_img.pop();

          for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
              for (int k = -1; k <= 1; k++) {
                int new_row = row + i;
                int new_col = col + j;
                int new_pic = pic + k;

                // index out of range
                if (new_col < 0 || new_row < 0 || new_pic < 1 ||
                    new_col >= edge_images[0].cols ||
                    new_row >= edge_images[0].rows ||
                    new_pic >= edge_images.size() - 1) {
                  continue;
                }

                if (edge_images[new_pic].at<uint8_t>(new_row, new_col) == 127) {
                  edge_images[new_pic].at<uint8_t>(new_row, new_col) = 255;
                  candidate_col.push(new_col);
                  candidate_row.push(new_row);
                  candidate_img.push(new_pic);
                }
              }
            }
          }
        }
      }
    }
  }

  for (size_t img_i = 1; img_i < edge_images.size() - 1; img_i++) {
    for (size_t i = 0; i < edge_images[img_i].rows; i++) {
      for (size_t j = 0; j < edge_images[img_i].cols; j++) {
        if (edge_images[img_i].at<uint8_t>(i, j) != 255) {
          edge_images[img_i].at<uint8_t>(i, j) = 0;
        }
      }
    }
  }
}
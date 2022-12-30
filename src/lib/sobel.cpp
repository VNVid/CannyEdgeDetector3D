#include <sobel.h>

SobelOperator::SobelOperator(std::vector<cv::Mat>& images, double coef)
    : images_(images) {
  for (cv::Mat img : images_) {
    gradient_.push_back(cv::Mat::zeros(img.size(), CV_32SC1));
    interpolated_gradient_.push_back(
        std::make_pair(cv::Mat::zeros(img.size(), CV_32SC1),
                       cv::Mat::zeros(img.size(), CV_32SC1)));
    grad_dir_x_.push_back(cv::Mat::zeros(img.size(), CV_32SC1));
    grad_dir_y_.push_back(cv::Mat::zeros(img.size(), CV_32SC1));
    grad_dir_z_.push_back(cv::Mat::zeros(img.size(), CV_32SC1));
  }
  for (size_t i = 0; i < images_.size(); i++) {
    cv::Mat prev;
    cv::Mat prev_prev;
    cv::Mat next;
    cv::Mat next_next;
    if (i == 0) {
      prev = images_[i];
      prev_prev = images_[i];
    } else {
      cv::Mat delta;
      cv::subtract(images_[i], images_[i - 1], delta);
      cv::addWeighted(images_[i], 1, delta, -coef, 0, prev);
      cv::addWeighted(images_[i], 1, delta, -coef * 2, 0, prev_prev);
    }

    if (i == images_.size() - 1) {
      next = images_[i];
      next_next = images_[i];
    } else {
      cv::Mat delta;
      cv::subtract(images_[i + 1], images_[i], delta);
      cv::addWeighted(images_[i], 1, delta, coef, 0, next);
      cv::addWeighted(images_[i], 1, delta, 2 * coef, 0, next_next);
    }
    std::vector<cv::Mat> neighbours;
    neighbours.push_back(prev_prev);
    neighbours.push_back(prev);
    neighbours.push_back(next);
    neighbours.push_back(next_next);
    interpolated_images_.push_back(neighbours);
  }
}

void SobelOperator::Count() {
  SobelFeldmanFilters filter = SobelFeldmanFilters();

  for (size_t img_i = 0; img_i < images_.size(); img_i++) {
    // while proccessing current image (img) also consider the previous and
    // the next ones, but with interpolation
    cv::Mat img = images_[img_i];
    cv::Mat prev_img = interpolated_images_[img_i][1];
    cv::Mat next_img = interpolated_images_[img_i][2];
    cv::Mat prev_prev_img = interpolated_images_[img_i][0];
    cv::Mat next_next_img = interpolated_images_[img_i][3];

    for (size_t i = 1; i < img.rows - 1; i++) {
      for (size_t j = 1; j < img.cols - 1; j++) {
        cv::Mat part = img(cv::Rect(j - 1, i - 1, 3, 3));
        cv::Mat prev_part = prev_img(cv::Rect(j - 1, i - 1, 3, 3));
        cv::Mat next_part = next_img(cv::Rect(j - 1, i - 1, 3, 3));
        part.convertTo(part, CV_32SC1);
        prev_part.convertTo(prev_part, CV_32SC1);
        next_part.convertTo(next_part, CV_32SC1);

        // counting gradients along directions
        int Gx = cv::sum(prev_part.mul(filter.xdir.kernel1)).val[0] +
                 cv::sum(part.mul(filter.xdir.kernel2)).val[0] +
                 cv::sum(next_part.mul(filter.xdir.kernel3)).val[0];
        int Gy = cv::sum(prev_part.mul(filter.ydir.kernel1)).val[0] +
                 cv::sum(part.mul(filter.ydir.kernel2)).val[0] +
                 cv::sum(next_part.mul(filter.ydir.kernel3)).val[0];
        int Gz = cv::sum(prev_part.mul(filter.zdir.kernel1)).val[0] +
                 cv::sum(part.mul(filter.zdir.kernel2)).val[0] +
                 cv::sum(next_part.mul(filter.zdir.kernel3)).val[0];

        // counting the gradient magnitude
        gradient_[img_i].at<int32_t>(i, j) =
            (int32_t)(sqrt(Gx * Gx + Gy * Gy + Gz * Gz));

        // calculating the gradient's direction
        // in means of pixels
        double best_cos = 0;
        for (int dx = -1; dx <= 1; dx++) {
          for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
              if (dx == 0 && dy == 0 && dz == 0) continue;
              double dot_product = dx * Gx + dy * Gy + dz * Gz;
              double norm = sqrt(dx * dx + dy * dy + dz * dz) *
                            sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
              if (std::abs(dot_product / norm) > best_cos) {  // closer to 1
                best_cos = std::abs(dot_product / norm);
                grad_dir_x_[img_i].at<int32_t>(i, j) = dx;
                grad_dir_y_[img_i].at<int32_t>(i, j) = dy;
                grad_dir_z_[img_i].at<int32_t>(i, j) = dz;
              }
            }
          }
        }

        // counting gradients for prev
        part = prev_img(cv::Rect(j - 1, i - 1, 3, 3));
        prev_part = prev_prev_img(cv::Rect(j - 1, i - 1, 3, 3));
        next_part = img(cv::Rect(j - 1, i - 1, 3, 3));
        part.convertTo(part, CV_32SC1);
        prev_part.convertTo(prev_part, CV_32SC1);
        next_part.convertTo(next_part, CV_32SC1);

        Gx = cv::sum(prev_part.mul(filter.xdir.kernel1)).val[0] +
             cv::sum(part.mul(filter.xdir.kernel2)).val[0] +
             cv::sum(next_part.mul(filter.xdir.kernel3)).val[0];
        Gy = cv::sum(prev_part.mul(filter.ydir.kernel1)).val[0] +
             cv::sum(part.mul(filter.ydir.kernel2)).val[0] +
             cv::sum(next_part.mul(filter.ydir.kernel3)).val[0];
        Gz = cv::sum(prev_part.mul(filter.zdir.kernel1)).val[0] +
             cv::sum(part.mul(filter.zdir.kernel2)).val[0] +
             cv::sum(next_part.mul(filter.zdir.kernel3)).val[0];

        // counting the gradient magnitude
        interpolated_gradient_[img_i].first.at<int32_t>(i, j) =
            (int32_t)(sqrt(Gx * Gx + Gy * Gy + Gz * Gz));

        // counting gradients for next
        part = next_img(cv::Rect(j - 1, i - 1, 3, 3));
        prev_part = img(cv::Rect(j - 1, i - 1, 3, 3));
        next_part = next_next_img(cv::Rect(j - 1, i - 1, 3, 3));
        part.convertTo(part, CV_32SC1);
        prev_part.convertTo(prev_part, CV_32SC1);
        next_part.convertTo(next_part, CV_32SC1);

        Gx = cv::sum(prev_part.mul(filter.xdir.kernel1)).val[0] +
             cv::sum(part.mul(filter.xdir.kernel2)).val[0] +
             cv::sum(next_part.mul(filter.xdir.kernel3)).val[0];
        Gy = cv::sum(prev_part.mul(filter.ydir.kernel1)).val[0] +
             cv::sum(part.mul(filter.ydir.kernel2)).val[0] +
             cv::sum(next_part.mul(filter.ydir.kernel3)).val[0];
        Gz = cv::sum(prev_part.mul(filter.zdir.kernel1)).val[0] +
             cv::sum(part.mul(filter.zdir.kernel2)).val[0] +
             cv::sum(next_part.mul(filter.zdir.kernel3)).val[0];

        // counting the gradient magnitude
        interpolated_gradient_[img_i].second.at<int32_t>(i, j) =
            (int32_t)(sqrt(Gx * Gx + Gy * Gy + Gz * Gz));
      }
    }
  }

  counted_ = true;
}
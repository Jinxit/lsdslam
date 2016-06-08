#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "map_range.hpp"

inline void show(const std::string& title, const Image& image, bool normalize = false,
                 float multiplier = 1.0f)
{
    cv::Mat cv_image;
    cv::eigen2cv(image, cv_image);
    if (normalize)
        cv::normalize(cv_image, cv_image, 1, 0, cv::NORM_INF);
    //std::cout << "show " << image(0, 0) << " " << cv_image.at<float>(0, 0) << std::endl;
    cv::imshow(title, cv_image * multiplier);
}

inline void show_rgb(const std::string& title, const Image& r, const Image& g, const Image& b,
                     int height, int width, bool normalize = false)
{
    cv::Mat cv_r, cv_g, cv_b;
    cv::eigen2cv(r, cv_r);
    cv::eigen2cv(g, cv_g);
    cv::eigen2cv(b, cv_b);

    if (normalize)
    {
        cv::normalize(cv_r, cv_r, 1, 0, cv::NORM_INF);
        cv::normalize(cv_g, cv_g, 1, 0, cv::NORM_INF);
        cv::normalize(cv_b, cv_b, 1, 0, cv::NORM_INF);
    }

    cv::Mat cv_image;
    cv::merge(std::vector<cv::Mat>{0*cv_b, cv_g, cv_r}, cv_image);
    cv::resize(cv_image, cv_image, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::imshow(title, cv_image);
}

inline void show_rainbow(const std::string& title, const studd::two<Image>& image,
                         const Image& base)
{
    auto height = image[0].rows();
    auto width = image[0].cols();

    cv::Mat h = cv::Mat::zeros(height, width, CV_32F);;
    cv::Mat s = cv::Mat::ones(height, width, CV_32F);
    cv::Mat v = cv::Mat::ones(height, width, CV_32F);

    cv::Mat r = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat g = cv::Mat::zeros(height, width, CV_32F);
    cv::Mat b = cv::Mat::zeros(height, width, CV_32F);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (image[1](y, x) < 0 || x < 5 || x > width - 5 || y < 5 || y > height - 5
                || image[0](y, x) == -1.0f)
            {
                v.at<float>(y, x) = 0;
            }
            else if (image[0](y, x) != 0)
            {
                h.at<float>(y, x) = std::abs(image[0](y, x));
            }
            if (image[1](y, x) > 0)
            {
                r.at<float>(y, x) = interpolate<float>(0, 1, image[1](y, x));
                g.at<float>(y, x) = 1.0f - r.at<float>(y, x);
            }
        }
    }
    cv::normalize(h, h, 255, 0, cv::NORM_INF);
    cv::Mat rainbow_image;
    cv::merge(std::vector<cv::Mat>{h, s, v}, rainbow_image);
    cv::cvtColor(rainbow_image, rainbow_image, CV_HSV2RGB);

    cv::Mat variance_image;
    cv::merge(std::vector<cv::Mat>{b, g, r}, variance_image);

    cv::imshow(title, rainbow_image);
    cv::imshow(title + "_var", variance_image);
}

inline void show_residuals(const std::string& title,
                           const Image& new_image, const Image& ref_image,
                           const sparse_gaussian& sparse_inverse_depth,
                           const sparse_gaussian& warped,
                           int height, int width, int magnification = 4)
{
    auto pyr = new_image.rows() / height;

    Image warped_image = Image::Zero(height, width);
    Image warped_image_inv_depth = Image::Zero(height, width);
    Image warped_image_variance = -Image::Ones(height, width);
    Image mask = Image::Zero(height, width);
    for (size_t j = 0; j < warped.size(); j++)
    {
        float x = float(warped[j].first.x()) / pyr;
        float y = float(warped[j].first.y()) / pyr;
        Eigen::Vector2f p(x, y);

        if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
        {
            int sx = sparse_inverse_depth[j].first.x() / pyr;
            int sy = sparse_inverse_depth[j].first.y() / pyr;

            if (sx > 0 && sx < width - 1 && sy > 0 && sy < height - 1)
            {
                gaussian warped_pixel = warped[j].second
                                      ^ gaussian(interpolate(warped_image_inv_depth, p),
                                                 interpolate(warped_image_variance, p));
                if (warped[j].second.mean > warped_image_inv_depth(sy, sx))
                {
                    if (x < width && y < height)
                        warped_image(sy, sx) = interpolate(new_image, p * pyr);
                }
                warped_image_inv_depth(sy, sx) = warped_pixel.mean;
                warped_image_variance(sy, sx) = warped_pixel.variance;
                mask(sy, sx) = 1;
            }
        }
    }

    Image resized_ref(height, width);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            resized_ref(y, x) = ref_image(y * pyr, x * pyr);
        }
    }

    Image residuals = (resized_ref - warped_image).cwiseAbs().cwiseProduct(mask);

    cv::Mat img = cv::Mat::zeros(height * magnification, width * magnification, CV_8UC3);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int i = 0; i < magnification; i++)
            {
                for (int j = 0; j < magnification; j++)
                {
                    auto col = resized_ref(y, x);
                    col *= 0;
                    img.at<cv::Vec3b>(y * magnification + i, x * magnification + j)
                        = cv::Vec3b(col * 255, col * 255, col * 255 + residuals(y, x) * 255.0);
                }
            }
        }
    }

    cv::imshow(title, img);
}


inline void show_residuals(const std::string& title, const Eigen::Matrix3f& intrinsic,
                           const Image& new_image, const Image& ref_image,
                           const sparse_gaussian& sparse_inverse_depth,
                           const Sophus::SE3f& transform,
                           int height, int width, int magnification = 4)
{
    show_residuals(title, new_image, ref_image, sparse_inverse_depth,
                   warp(sparse_inverse_depth, intrinsic, transform), height, width,
                   magnification);

}

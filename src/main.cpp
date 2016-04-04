#include <iostream>
#include <string>
#include <utility>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/eigen.hpp"

#include "dynamic_map.hpp"
#include "two.hpp"
#include "square.hpp"
#include "se3.hpp"
#include "eigen_utils.hpp"
#include "loader.hpp"

constexpr int max_pyramid = 4;

template<int Height, int Width>
using Matrix = Eigen::Matrix<float, Height, Width>;
using Image = Eigen::MatrixXf;

void show(const std::string& title, const Image& image, bool normalize = false,
          float multiplier = 1.0f)
{
    cv::Mat cv_image;
    cv::eigen2cv(image, cv_image);
    if (normalize)
        cv::normalize(cv_image, cv_image, 1, 0, cv::NORM_INF);
    //std::cout << "show " << image(0, 0) << " " << cv_image.at<float>(0, 0) << std::endl;
    cv::imshow(title, cv_image * multiplier);
}

void show(const std::string& title, const studd::two<Image>& image, bool normalize = false,
          float multiplier = 1.0f)
{
    cv::Mat g;
    cv::Mat b;
    cv::Mat r = cv::Mat::zeros(image[0].rows(), image[0].cols(), CV_32F);
    cv::eigen2cv(image[0], g);
    cv::eigen2cv(image[1], b);
    for (int y = 0; y < image[0].rows(); y++)
    {
        for (int x = 0; x < image[0].cols(); x++)
        {
            if (b.at<float>(y, x) < 0)
            {
                b.at<float>(y, x) = 0;
                r.at<float>(y, x) = 1;
            }
        }
    }
    if (normalize)
    {
        cv::normalize(g, g, 1, 0, cv::NORM_INF);
        cv::normalize(b, b, 1, 0, cv::NORM_INF);
    }
    cv::Mat cv_image;
    cv::merge(std::vector<cv::Mat>{b, g, r}, cv_image);
    //std::cout << "show " << image(0, 0) << " " << cv_image.at<float>(0, 0) << std::endl;
    cv::imshow(title, cv_image * multiplier);
}

void show_rainbow(const std::string& title, const studd::two<Image>& image)
{
    auto height = image[0].rows();
    auto width = image[0].cols();

    cv::Mat h = cv::Mat::zeros(height, width, CV_32F);;
    cv::Mat s = cv::Mat::ones(height, width, CV_32F);
    cv::Mat v = cv::Mat::ones(height, width, CV_32F);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (image[1](y, x) < 0 || x < 5 || x > width - 5 || y < 5 || y > height - 5)
            {
                v.at<float>(y, x) = 0;
            }
            else if (image[0](y, x) != 0)
            {
                h.at<float>(y, x) = 1.0 / image[0](y, x);
            }
        }
    }
    cv::normalize(h, h, 180, 0, cv::NORM_INF);
    cv::Mat cv_image;
    cv::merge(std::vector<cv::Mat>{h, s, v}, cv_image);
    cv::cvtColor(cv_image, cv_image, CV_HSV2RGB);
    cv::imshow(title, cv_image);
}

std::vector<cv::Point2f> fast_find(const cv::Mat& image)
{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> points;
    
    // use FAST to find features
    auto fast = cv::FastFeatureDetector::create(20);
    fast->detect(image, keypoints);
    cv::KeyPoint::convert(keypoints, points);

    return points;
}

studd::two<std::vector<cv::Point2f>> klt_track_points(const Image& _new_image,
                                                      const Image& _ref_image)
{
    cv::Mat new_image;
    cv::Mat ref_image;
    cv::eigen2cv(_new_image, new_image);
    cv::eigen2cv(_ref_image, ref_image);
    new_image.convertTo(new_image, CV_8UC1, 255);
    ref_image.convertTo(ref_image, CV_8UC1, 255);

    auto points1 = fast_find(ref_image);
    auto points2 = fast_find(new_image);

    // Kanade-Lucas-Tomasi tracking
    std::vector<float> err;
    std::vector<uchar> status;

    cv::calcOpticalFlowPyrLK(ref_image, new_image, points1, points2, status, err);

    studd::two<std::vector<cv::Point2f>> out_points;
    out_points[0].reserve(status.size());
    out_points[1].reserve(status.size());
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i] != 0 && points2[i].x >= 0 && points2[i].y >= 0)
        {
            out_points[0].push_back(points1[i]);
            out_points[1].push_back(points2[i]);
        }
    }

    return out_points;
}

std::pair<Eigen::Affine3f, Eigen::Matrix3f> keypoint_tracking(const Image& new_image,
                                                              const Image& ref_image,
                                                              const Eigen::Matrix3f& intrinsic)
{
    auto points = klt_track_points(new_image, ref_image);

    auto pp = cv::Point2f(intrinsic(0, 2), intrinsic(1, 2));
    // RANSAC
    auto fundamental = cv::findFundamentalMat(points[0], points[1], intrinsic(1, 1));
    auto essential = cv::findEssentialMat(points[0], points[1], intrinsic(1, 1), pp);

    cv::Mat rotation, translation;

    cv::recoverPose(essential, points[0], points[1], rotation, translation,
                    intrinsic(1, 1), pp);

    Eigen::Affine3f pose;
    Eigen::Matrix3f fundamental_eigen;
    Eigen::Matrix3f rotation_eigen;
    Eigen::Vector3f translation_eigen;
    cv::cv2eigen(fundamental, fundamental_eigen);
    cv::cv2eigen(rotation, rotation_eigen);
    pose.matrix().topLeftCorner<3, 3>() = rotation_eigen;
    cv::cv2eigen(translation, translation_eigen);
    pose.matrix().topRightCorner<3, 1>() = translation_eigen;
    return {pose, fundamental_eigen};
}

struct epiline
{
    epiline(const Eigen::Vector2i point, const Eigen::Vector3f line)
        : point(point), line(line) { };

    Eigen::Vector2i point;
    Eigen::Vector3f line;
};

epiline generate_epiline(const Eigen::Vector2i& p, const Eigen::Matrix3f& fundamental)
{
    Eigen::Vector3f line = fundamental * Eigen::Vector3f(p.x(), p.y(), 1);
    auto nu = studd::square(line[0]) + studd::square(line[1]);
    if (nu != 0)
    {
        line /= std::sqrt(nu);
    }
    return epiline(p, line);
}

std::vector<epiline> generate_epilines(const Image& new_image,
                                       const Image& ref_image,
                                       const Eigen::Matrix3f& fundamental)
{
    std::vector<epiline> epilines;
    epilines.reserve(new_image.rows() * new_image.cols());
    for (int y = 0; y < new_image.rows(); y++)
    {
        for (int x = 0; x < new_image.cols(); x++)
        {
            epilines.emplace_back(generate_epiline(Eigen::Vector2i(x, y), fundamental));
        }
    }

    return epilines;
}

Eigen::Vector2f get_epipole(const Eigen::Affine3f& transform, const Eigen::Matrix3f intrinsic)
{
    Eigen::Vector3f homogenous = intrinsic * transform.translation();
    return homogenous.topLeftCorner<2, 1>() / homogenous[2];
}

studd::two<Eigen::Vector2f> epiline_limits(Eigen::Vector3f epiline, int height, int width)
{
    auto fx = [&](float x) {
        return -(epiline[0] * x + epiline[2]) / epiline[1];
    };
    auto fy = [&](float y) {
        return -(epiline[1] * y + epiline[2]) / epiline[0];
    };
    float x0 = 0;
    float y0 = fx(0);
    if (y0 < 0)
    {
        y0 = 0;
        x0 = fy(y0);
    }
    else if (y0 > height)
    {
        y0 = height;
        x0 = fy(y0);
    }

    float x1 = width;
    float y1 = fx(width);
    if (y1 < 0)
    {
        y1 = 0;
        x1 = fy(y1);
    }
    else if (y1 > height)
    {
        y1 = height;
        x1 = fy(y1);
    }

    return studd::make_two(Eigen::Vector2f(x0, y0), Eigen::Vector2f(x1, y1));
}

studd::two<Image> disparity_epiline_ssd(
    const Image& new_image, const Image& ref_image,
    const Eigen::Matrix3f& fundamental, const std::vector<epiline>& epilines,
    const Eigen::Vector2f& epipole)
{
    constexpr float epiline_sample_distance = 5.0f;
    constexpr size_t num_epiline_samples = 5; // must be odd
    constexpr int half_epiline_samples = num_epiline_samples / 2;

    auto height = new_image.rows();
    auto width = new_image.cols();

    studd::two<Image> disparity = studd::make_two(Image::Zero(height, width),
                                                  Image::Zero(height, width));
    cv::Mat disps = cv::Mat::zeros(height, width, CV_8UC3);

    auto safe = [&](const Eigen::Vector2f& p) {
        return p.x() >= 0 && p.y() >= 0 && p.x() < width && p.y() < height;
    };

    auto ssd = [&](const std::array<float, num_epiline_samples> target,
                   Eigen::Vector2f p, const Eigen::Vector2f& dp) {
        float total = 0;
        for (size_t i = 0; i < num_epiline_samples; i++)
        {
            if (!safe(p))
                total += 1000;
            else
                total += studd::square(target[i] - new_image(p.y(), p.x()));

            p += dp;
        }

        return total;
    };

    float min_disp = 10000000;
    float max_disp = 0;
    float avg_disp = 0;

    int i = 0;
    for (auto&& epi : epilines)
    {
        //std::cout << "line " << i << " out of " << epilines.size() << std::endl;
        i++;
        Eigen::Vector2f p0, p1;
        std::tie(p0, p1) = epiline_limits(epi.line, height, width);

        std::array<float, num_epiline_samples> target{0};
        auto p = p0;
        Eigen::Vector2f dp = (p1 - p0).normalized() * epiline_sample_distance;

        // set up target values;
        for (int i = 0; i < int(num_epiline_samples); i++)
        {
            Eigen::Vector2f point = epi.point.cast<float>() + dp * float(i - half_epiline_samples);

            if (safe(point))
                target[i] = float(ref_image(point.y(), point.x()));
        }

        // find distance for n steps
        Eigen::Vector2f dp_total = dp * int(num_epiline_samples - 1);

        float min_ssd = std::numeric_limits<float>::infinity();
        Eigen::Vector2f min_p(-1000, -1000);
        // keep going for the rest of the line
        while (safe(p + dp_total))
        {
            if (safe(p + dp_total))
            {
                float current_ssd = ssd(target, p, dp);

                if (current_ssd < min_ssd)
                {
                    min_ssd = current_ssd;
                    min_p = p + dp_total / 2;
                }
            }

            p += dp;
        }

        if (min_p.x() == -1000)
        {
            continue;
        }

        disparity[0](epi.point.y(), epi.point.x()) = min_p.x() - epi.point.x();
        disparity[1](epi.point.y(), epi.point.x()) = min_p.y() - epi.point.y();

        auto disp_n = (epi.point.cast<float>() - min_p).norm();
        if (disp_n > max_disp)
            max_disp = disp_n;
        if (disp_n < min_disp)
            min_disp = disp_n;
        avg_disp += disp_n;
    }
    avg_disp /= epilines.size();

    return disparity;
}

Image random_inverse_depth(int height, int width)
{
    static std::mt19937 gen;
    static std::uniform_real_distribution<float> dist(0.1, 2);
    Image depth = Image::Zero(height, width);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            depth(y, x) = 1.0 / dist(gen);
        }
    }

    return depth;
}

struct gaussian
{
    gaussian(float mean, float variance)
        : mean(mean), variance(variance) { };

    friend gaussian operator^(const gaussian& lhs, const gaussian& rhs)
    {
        if (rhs.variance < 0)
        {
            return lhs;
        }
        if (lhs.variance < 0)
        {
            return rhs;
        }

        if (std::abs(lhs.mean - rhs.mean) > lhs.variance * 2)
        {
            if (lhs.mean > rhs.mean)
            {
                return lhs;
            }
            else
            {
                return rhs;
            }
        }

        auto mean = (lhs.variance * rhs.mean + rhs.variance * lhs.mean)
                    / (lhs.variance + rhs.variance);
        auto variance = (lhs.variance * rhs.variance) / (lhs.variance + rhs.variance);
        return gaussian(mean, variance);
    }

    friend std::ostream& operator<< (std::ostream& stream, const gaussian& self) {
        stream << self.mean << " " << self.variance;
        return stream;
    }

    float mean;
    float variance;
};

gaussian geometric_disparity(const studd::two<Image>& gradient,
                             const epiline& epi, int x, int y,
                             const Eigen::Vector2f& epipole)
{
    constexpr float sigma_l = 0.1f;

    Eigen::Vector2f g(gradient[0](y, x), gradient[1](y, x));
    if (g.x() == 0 && g.y() == 0)
        return gaussian(0, -1);
    g.normalize();

    // semi-dense visual odometry, equation 5 and 6
    auto g_dot_l = g.x() * epi.line[0] + g.y() * epi.line[1];
    auto disparity = (g.x() * (x - epipole.x()) + g.y() * (y - epipole.y())) / g_dot_l;
    auto variance = (sigma_l * sigma_l) / (g_dot_l * g_dot_l);
    return gaussian(disparity, variance);
}

gaussian photometric_disparity(const Image& new_image,
                               const Image& ref_image,
                               const studd::two<Image>& gradient,
                               const epiline& epi, int x, int y,
                               const Eigen::Vector2f& disparity_0,
                               const Eigen::Vector2f& epipole)
{
    constexpr float sigma_i = 0.01f;

    Eigen::Vector2f g(gradient[0](y, x), gradient[1](y, x));
    auto g_p = g.x() * epi.line[0] + g.y() * epi.line[1];
    if (g_p == 0)
        return gaussian(0, -1);

    // semi-dense visual odometry, equation 8 and 9
    Eigen::Vector2f lambda_0 = Eigen::Vector2f(disparity_0[0] + epi.point.x(),
                                               disparity_0[1] + epi.point.y());
    auto disparity = disparity_0.norm()
                   + (ref_image(y, x) - new_image(lambda_0.y(), lambda_0.x())) / g_p;
    auto variance = (sigma_i * sigma_i) / (g_p * g_p);
    return gaussian(disparity, variance);
}

studd::two<Image> regularize_depth(const Image& inverse_depth, const Image& variance)
{
    auto height = inverse_depth.rows();
    auto width = inverse_depth.cols();

    Image output_inv_depth = Image::Zero(height, width);
    Image output_variance = Image::Zero(height, width);

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int num_added = 0;

            auto original_inv_depth = inverse_depth(y, x);
            auto original_variance = variance(y, x);
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    auto adjacent_inv_depth = inverse_depth(y + dy, x + dx);
                    auto adjacent_variance = variance(y + dy, x + dx);

                    if (std::abs(adjacent_inv_depth - original_inv_depth) > original_variance * 2
                     || adjacent_variance < 0)
                        continue;

                    output_inv_depth(y, x) += adjacent_inv_depth;
                    num_added++;
                }
            }

            if (num_added > 0)
            {
                output_inv_depth(y, x) /= num_added;
                output_variance(y, x) = original_variance;
            }
            else
            {
                output_inv_depth(y, x) = 0;
                output_variance(y, x) = -1;
            }
        }
    }

    return {output_inv_depth, output_variance};
}

struct pixel
{
    pixel(float inverse_depth, float variance, float x, float y)
        : inverse_depth(inverse_depth), variance(variance), x(x), y(y) { };

    float inverse_depth = 0;
    float variance = 0;
    float x = 0;
    float y = 0;
};

// Odometry from RGB-D Cameras for Autonomous Quadrocopters (eq 4.22 onwards)
Matrix<1, 6> image_pose_jacobian(const studd::two<Image>& gradient,
                                 float x, float y, float inverse_depth,
                                 const Eigen::Matrix3f& intrinsic, int pyr)
{
    Matrix<1, 2> J_I = Matrix<1, 2>::Zero();
    J_I(0, 0) = gradient[0](y / pyr, x / pyr);
    J_I(0, 1) = gradient[1](y / pyr, x / pyr);

    Matrix<2, 3> J_pi = Matrix<2, 3>::Zero();
    J_pi(0, 0) =  intrinsic(0, 0) * inverse_depth;
    J_pi(1, 1) =  intrinsic(1, 1) * inverse_depth;
    // verify these
    J_pi(0, 2) = -intrinsic(0, 0) * x * studd::square(inverse_depth);
    J_pi(1, 2) = -intrinsic(1, 1) * y * studd::square(inverse_depth);

    Matrix<3, 12> J_g = Matrix<3, 12>::Zero();
    for (int i = 0; i < 3; i++)
    {
        J_g(i, i    ) = x;
        J_g(i, i + 3) = y;
        J_g(i, i + 6) = 1.0 / inverse_depth;
        J_g(i, i + 9) = 1;
    }

    Matrix<12, 6> J_G = Matrix<12, 6>::Zero();
    J_G(1,  5) =  1;
    J_G(2,  4) = -1;
    J_G(3,  5) = -1;
    J_G(5,  3) =  1;
    J_G(6,  4) =  1;
    J_G(7,  3) = -1;
    J_G(9,  0) =  1;
    J_G(10, 1) =  1;
    J_G(11, 2) =  1;

    return J_I * J_pi * J_g * J_G;
}

std::vector<pixel> warp(std::vector<pixel> sparse_inverse_depth, const Eigen::Matrix3f& intrinsic,
                        const Eigen::Affine3f& ksi)
{
    for (auto&& p : sparse_inverse_depth)
    {
        // pi_1^-1
        Eigen::Vector3f p2;
        p2[0] = (p.x - intrinsic(0, 2)) / (intrinsic(0, 0) * p.inverse_depth);
        p2[1] = (p.y - intrinsic(1, 2)) / (intrinsic(1, 1) * p.inverse_depth);
        p2[2] = 1.0 / p.inverse_depth;

        // T
        p2 = ksi * p2;

        // pi_2
        p.x = p2[0] * intrinsic(0, 0) / p2[2] + intrinsic(0, 2);
        p.y = p2[1] * intrinsic(1, 1) / p2[2] + intrinsic(1, 2);
        p.inverse_depth = 1.0 / p2[2];
    }

    return sparse_inverse_depth;
}

std::pair<Eigen::VectorXf, Eigen::DiagonalMatrix<float, Eigen::Dynamic>> residuals_and_weights(
    const std::vector<pixel>& sparse_inverse_depth, const std::vector<pixel>& warped,
    const Image& new_image, const Image& ref_image,
    const std::function<float(float)>& weighting)
{
    Eigen::VectorXf residuals = Eigen::VectorXf::Zero(sparse_inverse_depth.size());
    Eigen::DiagonalMatrix<float, Eigen::Dynamic> weights(sparse_inverse_depth.size());

    for (size_t i = 0; i < warped.size(); i++)
    {
        int x = warped[i].x;
        int y = warped[i].y;
        if (x >= 0 && x < new_image.cols() && y >= 0 && y < new_image.rows())
        {
            float ref_intensity = ref_image(sparse_inverse_depth[i].y,
                                            sparse_inverse_depth[i].x);
            float new_intensity = new_image(y, x);
            auto residual = new_intensity - ref_intensity;
            residuals(i) = studd::square(residual);
            weights.diagonal()[i] = weighting(residual);
        }
        else
        {
            weights.diagonal()[i] = 0;
        }
    }

    return {residuals, weights};
}

Eigen::Matrix3f fundamental_from_pose(const Eigen::Affine3f& pose, const Eigen::Matrix3f& intrinsic)
{
    Eigen::Vector3f e = intrinsic * pose.rotation().transpose() * pose.translation();
    return intrinsic.transpose().inverse() * pose.rotation() * skewed(e);
}

void plot_errors(const se3& id,
                 const std::vector<pixel>& sparse_inverse_depth,
                 const Image& new_image,
                 const Image& ref_image,
                 const Eigen::Matrix3f& intrinsic,
                 const std::function<float(float)>& weighting)
{
    static int file_counter = 0;

    auto height = new_image.rows();
    auto width = new_image.cols();

    auto calc_error = [&](const se3 ksi, bool save = false) {
        auto warped = warp(sparse_inverse_depth, intrinsic, ksi.exp());
        Image warped_image = Image::Zero(height, width);
        Image warped_image_inv_depth = Image::Zero(height, width);
        for (size_t i = 0; i < warped.size(); i++)
        {
            int x = warped[i].x;
            int y = warped[i].y;
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                int sx = sparse_inverse_depth[i].x;
                int sy = sparse_inverse_depth[i].y;

                if (warped_image_inv_depth(y, x) < warped[i].inverse_depth)
                {
                    warped_image_inv_depth(y, x) = warped[i].inverse_depth;
                    warped_image(y, x) = ref_image(sy, sx);
                }
            }
        }

        if (save)
        {
            cv::Mat cv_image;
            cv::eigen2cv(warped_image, cv_image);
            std::stringstream uh;
            uh << "warped/" << std::setfill('0') << std::setw(5) << file_counter << "_" << ksi << ".jpg";
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (cv_image.at<float>(y, x) == 0)
                        cv_image.at<float>(y, x) = std::abs(cv_image.at<float>(y, x) - new_image(y, x));
                }
            }
            cv::imwrite(uh.str(), cv_image * 255);
            file_counter++;
        }
        studd::two<Image> gradient = sobel(warped_image);
        Eigen::VectorXf residuals;
        Eigen::DiagonalMatrix<float, Eigen::Dynamic> weights;
        std::tie(residuals, weights) = residuals_and_weights(sparse_inverse_depth, warped,
                                                             new_image, ref_image, weighting);
        float error = ((residuals.transpose() * weights).dot(residuals)
                    / sparse_inverse_depth.size());
        return error;
    };

    auto lerp = [](float v0, float v1, float t) {
        return (1 - t) * v0 + t * v1;
    };
    /*for (float t = 0.0f; t <= 1.0f; t += 0.005f)
    {
        calc_error(se3(lerp(0, id.omega[0], t),
                       lerp(0, id.omega[1], t),
                       lerp(0, id.omega[2], t),
                       lerp(0, id.nu[0], t),
                       lerp(0, id.nu[1], t),
                       lerp(0, id.nu[2], t)), true);
    }*/
/*
    for (float val = -50.0f; val < 50.0f; val += 5.0f)
    {
        calc_error(id ^ se3(0, 0, 0, val, 0, 0), true);
    }
    for (float val = -50.0f; val < 50.0f; val += 5.0f)
    {
        calc_error(id ^ se3(0, 0, 0, 0, val, 0), true);
    }
    for (float val = -50.0f; val < 50.0f; val += 5.0f)
    {
        calc_error(id ^ se3(0, 0, 0, 0, 0, val), true);
    }
    for (float val = -0.15f; val < 0.15f; val += 0.005f)
    {
        calc_error(id ^ se3(val, 0, 0, 0, 0, 0), true);
    }
    for (float val = -0.15f; val < 0.15f; val += 0.005f)
    {
        calc_error(id ^ se3(0, val, 0, 0, 0, 0), true);
    }
    for (float val = -0.15f; val < 0.15f; val += 0.005f)
    {
        calc_error(id ^ se3(0, 0, val, 0, 0, 0), true);
    }
    exit(0);*/

    auto start = std::chrono::steady_clock::now();
    //std::cout << "errors" << std::endl;
    int count = 0;
    static std::ofstream f("errors.txt");
    static std::ofstream g("minima.txt");
    static std::ofstream h("starts.txt");
    static int val_count = 0;

    se3 min_tf = id;
    float min_error = calc_error(id);
    se3 tf;
    float error;
    bool found_smaller = false;
    for (float val = -0.4f; val < 0.4f; val += 0.04f)
    {
        std::cout << "warping " << val << std::endl;
        if (std::abs(val) < 0.0001)
        {
            g << val_count << std::endl;
        }

        tf = id ^ se3(0, 0, 0, val * 2, 0, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.nu[0]) < 1e-2)
            h << 0 << " " << val_count << std::endl;

        tf = id ^ se3(0, 0, 0, 0, val * 2, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.nu[1]) < 1e-2)
            h << 1 << " " << val_count << std::endl;

        tf = id ^ se3(0, 0, 0, 0, 0, val * 2);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.nu[2]) < 1e-2)
            h << 2 << " " << val_count << std::endl;

        tf = id ^ se3(val, 0, 0, 0, 0, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.omega[0]) < 1e-2)
            h << 3 << " " << val_count << std::endl;

        tf = id ^ se3(0, val, 0, 0, 0, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.omega[1]) < 1e-2)
            h << 4 << " " << val_count << std::endl;

        tf = id ^ se3(0, 0, val, 0, 0, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.omega[2]) < 1e-2)
            h << 5 << " " << val_count << std::endl;

        f << std::endl;
        count += 6;
        val_count++;
    }
    if (!found_smaller)
    {
        std::cout << "finished search, saving images" << std::endl;
        std::cout << "ksi: " << min_tf << std::endl;
        for (float t = 0.0f; t <= 1.0f; t += 0.005f)
        {
            calc_error(se3(lerp(0, id.omega[0], t),
                           lerp(0, id.omega[1], t),
                           lerp(0, id.omega[2], t),
                           lerp(0, id.nu[0], t),
                           lerp(0, id.nu[1], t),
                           lerp(0, id.nu[2], t)), true);
        }
    }
    else
    {
        std::cout << "found smaller! " << min_error << std::endl;
        return plot_errors(min_tf, sparse_inverse_depth, new_image, ref_image, intrinsic, weighting);
    }
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    //std::cout << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
    //std::cout << "for " << count << " warps" << std::endl;
}

Eigen::Affine3f photometric_tracking(const std::vector<std::vector<pixel>>& sparse_inverse_depth,
                                     const Image& new_image,
                                     const Image& ref_image,
                                     const Eigen::Matrix3f& intrinsic,
                                     const std::function<float(float)>& weighting)
{
    constexpr size_t max_iterations = 10;
    constexpr float epsilon = 1e-9;

    auto height = new_image.rows();
    auto width = new_image.cols();

    std::cout << 1 << std::endl;

    se3 ksi;
    se3 delta_ksi;
    float last_error = std::numeric_limits<float>::infinity();

    int pi = -1;
    for (int pyr = max_pyramid; pyr > 0; pyr /= 2)
    {
        pi++;
        std::cout << std::endl << "RESOLUTION " << (width / pyr) << " x " << (height / pyr) << std::endl;
        if (sparse_inverse_depth[pi].size() == 0)
            continue;

        for (size_t i = 0; i < max_iterations; i++)
        {
            se3 new_ksi;
            if (i != 0)
            {
                if (delta_ksi.omega[0] != delta_ksi.omega[0])
                {
                    std::cerr << "erroring" << std::endl;
                    std::cerr << sparse_inverse_depth[pi].size() << std::endl;
                    plot_errors(ksi, sparse_inverse_depth[pi], new_image, ref_image, intrinsic, weighting);
                    exit(1);
                }
                new_ksi = ksi ^ delta_ksi;
                std::cout << "new ksi: " << std::endl << new_ksi << std::endl;
            }
            std::cout << 2 << std::endl;

            auto warped = warp(sparse_inverse_depth[pi], intrinsic, new_ksi.exp());
            std::cout << 2.5 << std::endl;

            // TODO: this is dumb as fuck, how can we do gradients better?
            Image warped_image = Image::Zero(height / pyr, width / pyr);
            Image warped_image_inv_depth = Image::Zero(height / pyr, width / pyr);
            for (size_t i = 0; i < warped.size(); i++)
            {
                int x = warped[i].x / pyr;
                int y = warped[i].y / pyr;
                if (x >= 0 && x < width / pyr && y >= 0 && y < height / pyr)
                {
                    int sx = sparse_inverse_depth[pi][i].x;
                    int sy = sparse_inverse_depth[pi][i].y;

                    if (warped_image_inv_depth(y, x) < warped[i].inverse_depth)
                    {
                        warped_image_inv_depth(y, x) = warped[i].inverse_depth;
                        warped_image(y, x) = ref_image(sy, sx);
                    }
                }
            }
            std::cout << 3 << std::endl;

            Image ref_resized = Image::Zero(height / pyr, width / pyr);
            for (int y = 0; y < height; y += pyr)
            {
                for (int x = 0; x < width; x += pyr)
                {
                    ref_resized(y / pyr, x / pyr) = ref_image(y, x);
                }
            }
            studd::two<Image> gradient = sobel(warped_image);
            //show("warped", warped_image);
            //show("gradient x", gradient[0]);
            //show("gradient y", gradient[1]);
            //cv::waitKey(0);

            Eigen::VectorXf residuals;
            Eigen::DiagonalMatrix<float, Eigen::Dynamic> weights;
            std::tie(residuals, weights) = residuals_and_weights(sparse_inverse_depth[pi], warped,
                                                                 new_image, ref_image, weighting);
            std::cout << 4 << std::endl;

            float error = ((weights * residuals).dot(residuals)
                        / sparse_inverse_depth[pi].size());

            std::cout << "sparse_inverse_depth size = " << std::endl << sparse_inverse_depth[pi].size() << std::endl;
            // consider switching to (last_error - error) / last_error < eps, according to wiki
            //if (error > last_error)
            if (error - last_error > 1e-2)
            {
                std::cout << "BOOM" << std::endl;
                std::cout << "errorb: " << error << std::endl;
                std::cout << "errord: " << ((last_error - error) / last_error) << std::endl;
                //delta_ksi = transform();
                //new_ksi = transform(ksi);
                //continue;
                delta_ksi = se3();
                break;
                //exit(1);
            }
            std::cout << 4.5 << std::endl;
            if (error < epsilon)
            {
                std::cout << "error is zero yo" << std::endl;
                break;
            }

            Eigen::MatrixXf J = Eigen::MatrixXf(warped.size(), 6);
            for (size_t i = 0; i < sparse_inverse_depth[pi].size(); i++)
            {
                auto& p = sparse_inverse_depth[pi][i];
                J.row(i) = image_pose_jacobian(gradient, p.x, p.y, p.inverse_depth, intrinsic, pyr);
            }
            std::cout << 5 << std::endl;

            Eigen::MatrixXf J_t_weights = J.transpose() * weights;
            ksi = new_ksi;
            //delta_ksi = se3((J_t_weights * J).inverse() * J_t_weights * residuals);
            delta_ksi = se3((J_t_weights * J).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                            .solve(-J_t_weights * residuals).topLeftCorner<6, 1>());

            std::cout << 6 << std::endl;
            std::cout << "ksi: " << std::endl << ksi.omega << std::endl
                      << ksi.nu << std::endl;
            std::cout << "delta_ksi: " << std::endl << delta_ksi.omega << std::endl
                      << delta_ksi.nu << std::endl;
            std::cout << "new_ksi: " << std::endl << new_ksi.omega << std::endl
                      << new_ksi.nu << std::endl;
            std::cout << "error: " << error << std::endl;
            if (std::abs(error - last_error) < epsilon)
                break;

            last_error = error;
        }
        //plot_errors(ksi, sparse_inverse_depth[pi], new_image, ref_image, weighting);
    }
    plot_errors(ksi, sparse_inverse_depth[pi], new_image, ref_image, intrinsic, weighting);
    exit(0);

    std::cout << "returning " << std::endl << ksi << std::endl;
    std::cout << "exped " << std::endl << ksi.exp().matrix() << std::endl;
    return ksi.exp();
}

studd::two<Image> fuse_depth(const studd::two<Image>& lhs, const studd::two<Image>& rhs)
{
    studd::two<Image> output = studd::make_two(Image::Zero(lhs[0].rows(), lhs[0].cols()),
                                               Image::Zero(lhs[0].rows(), lhs[0].cols()));
    for (int y = 0; y < lhs[0].rows(); y++)
    {
        for (int x = 0; x < lhs[0].cols(); x++)
        {
            auto fused = gaussian(lhs[0](y, x), lhs[1](y, x)) ^ gaussian(rhs[0](y, x), rhs[1](y, x));
            output[0](y, x) = fused.mean;
            output[1](y, x) = fused.variance;
        }
    }
    return output;
}

int main(int argc, const char* argv[])
{
    auto data = loader("data/MH_01_easy/");

    unsigned int frame_skip = 3;

    auto height = data[0].left.rows();
    auto width = data[0].left.cols();

    Eigen::Affine3f pose;
    Eigen::Matrix3f fundamental;
    std::tie(pose, fundamental) = keypoint_tracking(data[frame_skip + 1].left, data[1].left,
                                                    data.get_calibration().left.intrinsic);
    std::cout << pose.matrix() << std::endl;
    auto epipole = get_epipole(pose, data.get_calibration().left.intrinsic);
    studd::two<Image> inverse_depth = studd::make_two(Image::Zero(height, width),
                                                       Image::Constant(height, width, -1));
    std::cout << "epipole: " << std::endl << epipole << std::endl;
    for (size_t i = frame_skip + 1; i < 50000; i += frame_skip)
    {
        auto new_image = data[i].left;
        auto ref_image = data[i - frame_skip].left;
        std::cerr << "tick " << i << std::endl;
        std::cout << "pose: " << std::endl << pose.matrix() << std::endl;
        std::cout << "fundamental: " << std::endl << fundamental << std::endl;
        std::cout << "m" << 0 << std::endl;
        std::cout << "i: " << i << std::endl;
        auto epilines = generate_epilines(new_image, ref_image, fundamental);
        cv::Mat epim = cv::Mat::zeros(height, width, CV_32FC3);
        cv::Mat r, g, b = cv::Mat::zeros(height, width, CV_32F);
        cv::eigen2cv(new_image, r);
        cv::eigen2cv(ref_image, g);
        cv::merge(std::vector<cv::Mat>({b, g, r}), epim);
        for (int j = 0; j < epilines.size(); j += 3723)
        {
            cv::line(epim, cv::Point2f(epilines[j].point.x(), epilines[j].point.y()), cv::Point2f(epipole.x(), epipole.y()), cv::Scalar(255, 255, 255));
            //cv::line(epim, cv::Point2f(epilines[j].point.x(), epilines[j].point.y()), cv::Point2f(epilines[j].point.x(), epilines[j].point.y()) + cv::Point2f(epilines[j].line.x(), epilines[j].line.y()) * 100, cv::Scalar(255, 255, 0));
        }
        //cv::imshow("epim new", epim);
        std::cout << "m" << 0.1 << std::endl;
        studd::two<Image> gradient = sobel(ref_image);
        std::cout << "m" << 1 << std::endl;

        studd::two<Image> disparity = disparity_epiline_ssd(new_image, ref_image,
                                                            fundamental, epilines, epipole);
        studd::two<Image> geometric = studd::make_two(Image::Zero(height, width),
                                                      Image::Zero(height, width));
        studd::two<Image> photometric = studd::make_two(Image::Zero(height, width),
                                                        Image::Zero(height, width));
        Image masked = new_image;
        std::cout << "m" << 2 << std::endl;
        size_t j = 0;
        size_t geo_wins = 0;
        size_t photo_wins = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                auto disp = Eigen::Vector2f{disparity[0](y, x), disparity[1](y, x)};
                auto geo = geometric_disparity(gradient, epilines[j], x, y, epipole);
                auto photo = photometric_disparity(new_image, ref_image,
                                                   gradient, epilines[j], x, y, disp, epipole);

                if (std::min(geo.variance, photo.variance) > 0.01)
                {
                    masked(y, x) = 0;
                    if (photo.variance < geo.variance)
                        photo_wins++;
                    else
                        geo_wins++;
                }
                else
                {
                    geometric[0](y, x) = std::abs(geo.mean);
                    geometric[1](y, x) = geo.variance;
                    photometric[0](y, x) = std::abs(photo.mean);
                    photometric[1](y, x) = photo.variance;
                }
                j++;
            }
        }

        std::cout << "m" << 3 << std::endl;
        // filter out bad candidates
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                auto geo_var = geometric[1](y, x);
                auto photo_var = photometric[1](y, x);
                if (std::min(geo_var, photo_var) > 0.01 || (geo_var == 0 && photo_var == 0))
                {
                    disparity[0](y, x) = 0;
                    disparity[1](y, x) = -1;
                }
                else
                {
                    // TODO: times alpha^2, figure out what alpha is
                    disparity[0](y, x) = std::hypot(disparity[0](y, x), disparity[1](y, x));
                    disparity[1](y, x) = geo_var + photo_var;
                }
            }
        }
        std::cout << "m" << 4 << std::endl;

        // 1 / Z = d / (b * f)
        float denum = ((pose.translation().norm()) * data.get_calibration().left.intrinsic(1, 1));
        //show_rainbow("before", inverse_depth);
        //show_rainbow("new", studd::make_two(disparity[0] / denum, disparity[1] / denum));
        inverse_depth = fuse_depth(inverse_depth,
                                   studd::make_two(disparity[0] / denum, disparity[1] / denum));
        //show_rainbow("after", inverse_depth);
        //cv::waitKey(0);

        inverse_depth = regularize_depth(inverse_depth[0], inverse_depth[1]);
        std::cout << "m" << 5 << std::endl;

        // TODO: outlier removal

        //show("photometric", photometric[1], true, 4);
        //show("geometric", geometric[1], true, 4);
        //show("gradient x", gradient[0], true);
        //show("gradient y", gradient[1], true);
        //show("disparity mean", disparity[0], true);
        //show("disparity variance", disparity[1], true, 4);
        //show("regularized depth mean", inverse_depth[0], true);
        //show("masked", masked);

        std::vector<std::vector<pixel>> sparse_inverse_depth;
        //sparse_inverse_depth.reserve(height * width);
        for (int pyr = max_pyramid; pyr > 0; pyr /= 2)
        {
            sparse_inverse_depth.emplace_back();
            for (int y = 0; y < height; y += pyr)
            {
                for (int x = 0; x < width; x += pyr)
                {
                    auto inv_mean = inverse_depth[0](y, x);
                    auto inv_var = inverse_depth[1](y, x);
                    if (inv_var >= 0 && inv_mean != 0)
                    {
                        sparse_inverse_depth.back().emplace_back(inv_mean, inv_var, x, y);
                    }
                }
            }
        }

        pose = photometric_tracking(sparse_inverse_depth,
                                    new_image, ref_image,
                                    data.get_calibration().left.intrinsic,
                                    [](float r) { return 1; });
        std::cout << pose.rotation() << std::endl << pose.translation() << std::endl;
        std::cout << fundamental << std::endl << epipole << std::endl;
        fundamental = fundamental_from_pose(pose, data.get_calibration().left.intrinsic);
        std::cout << fundamental << std::endl << epipole << std::endl;

        std::cout << "m" << 6 << std::endl;
        //cv::waitKey(1000);
    }
}

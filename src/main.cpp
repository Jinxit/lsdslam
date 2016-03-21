#include <iostream>
#include <string>
#include <utility>
#include <cmath>
#include <random>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

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

const Eigen::Vector2f focal_length = Eigen::Vector2f{254.32, 375.93};
const Eigen::Vector2f principal_point = Eigen::Vector2f{267.38, 231.59};
constexpr int width = 640;
constexpr int height = 480;
Eigen::Matrix3f K = Eigen::Matrix3f::Zero();

template<int Height, int Width>
using Matrix = Eigen::Matrix<float, Height, Width>;
using Image = Eigen::MatrixXf;

Image load_image(const unsigned int& id)
{
    auto name = std::to_string(id);
    name.insert(0, 5 - name.size(), '0');
    cv::Mat image = cv::imread("data/LSD_room/images/" + name + ".png", CV_LOAD_IMAGE_GRAYSCALE);

    Image output = Image::Zero(height, width);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            output(y, x) = image.at<unsigned char>(y, x) / 255.0;
        }
    }
    std::cout << "loaded " << id << std::endl;
    return output;
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
                                                              const Image& ref_image)
{
    auto points = klt_track_points(new_image, ref_image);

    auto pp = cv::Point2f(principal_point.x(), principal_point.y());
    // RANSAC
    auto fundamental = cv::findFundamentalMat(points[0], points[1], focal_length.y());
    auto essential = cv::findEssentialMat(points[0], points[1], focal_length.y(), pp);

    cv::Mat rotation, translation;

    cv::recoverPose(essential, points[0], points[1], rotation, translation,
                    focal_length.y(), pp);

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
    epilines.reserve(width * height);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            epilines.emplace_back(generate_epiline(Eigen::Vector2i(x, y), fundamental));
        }
    }

    return epilines;
}

Eigen::Vector2f get_epipole(const Eigen::Affine3f& transform)
{
    return (K * transform.translation()).topLeftCorner<2, 1>();
}

studd::two<Eigen::Vector2f> epiline_limits(Eigen::Vector3f epiline)
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

float interpolate(const Image& image, const Eigen::Vector2f& p)
{
    Eigen::Vector2i ip = p.cast<int>();
    Eigen::Vector2f dp = p - ip.cast<float>();

    float top_left     = (1.0 - dp.x()) * (1.0 - dp.y());
    float top_right    = dp.x()         * (1.0 - dp.y());
    float bottom_left  = (1.0 - dp.x()) * dp.y();
    float bottom_right = dp.x()         * dp.y();

    return image(ip.y(),     ip.x()    ) * top_left
         + image(ip.y(),     ip.x() + 1) * top_right
         + image(ip.y() + 1, ip.x()    ) * bottom_left
         + image(ip.y() + 1, ip.x() + 1) * bottom_right;
}


studd::two<Image> disparity_epiline_ssd(
    const Image& new_image, const Image& ref_image,
    const Eigen::Matrix3f& fundamental, const std::vector<epiline>& epilines,
    const Eigen::Vector2f& epipole)
{
    constexpr float epiline_sample_distance = 5.0f;
    constexpr size_t num_epiline_samples = 5; // must be odd
    constexpr int half_epiline_samples = num_epiline_samples / 2;

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
        std::tie(p0, p1) = epiline_limits(epi.line);

        std::array<float, num_epiline_samples> target{0};
        auto p = p0;
        Eigen::Vector2f dp = p1 - p0;
        dp = dp.normalized() * epiline_sample_distance;

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

Image random_inverse_depth()
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
    constexpr float sigma_i = 1.0f;

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
    Image output_inv_depth = Image::Zero(height, width);
    Image output_variance = Image::Zero(height, width);

    for (int y = 1; y < int(height - 1); y++)
    {
        for (int x = 1; x < int(width - 1); x++)
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
                                 float x, float y, float inverse_depth)
{
    Matrix<1, 2> J_I = Matrix<1, 2>::Zero();
    J_I(0, 0) = gradient[0](y, x);
    J_I(0, 1) = gradient[1](y, x);

    Matrix<2, 3> J_pi = Matrix<2, 3>::Zero();
    J_pi(0, 0) =  focal_length.x() * inverse_depth;
    J_pi(1, 1) =  focal_length.y() * inverse_depth;
    // verify these
    J_pi(0, 2) = -focal_length.x() * x * studd::square(inverse_depth);
    J_pi(1, 2) = -focal_length.y() * y * studd::square(inverse_depth);

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

std::vector<pixel> warp(std::vector<pixel> sparse_inverse_depth, const Eigen::Affine3f& ksi)
{
    for (auto&& p : sparse_inverse_depth)
    {
        // pi_1^-1
        Eigen::Vector3f p2;
        p2[0] = (p.x - principal_point.x()) / (focal_length.x() * p.inverse_depth);
        p2[1] = (p.y - principal_point.y()) / (focal_length.y() * p.inverse_depth);
        p2[2] = 1.0 / p.inverse_depth;

        // T
        p2 = ksi * p2;

        // pi_2
        p.x = p2[0] * focal_length.x() / p2[2] + principal_point.x();
        p.y = p2[1] * focal_length.y() / p2[2] + principal_point.y();
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
        if (x >= 0 && x < width && y >= 0 && y < height)
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

Eigen::Matrix3f skewed(const Eigen::Vector3f& x)
{
    Eigen::Matrix3f skewed;
    skewed <<    0, -x[2],  x[1],
              x[2],     0, -x[0],
             -x[1],  x[0],     0;
    return skewed;
}

Eigen::Vector3f unskewed(const Eigen::Matrix3f& m)
{
    Eigen::Vector3f unskewed;
    unskewed[0] = m(2, 1);
    unskewed[1] = m(0, 2);
    unskewed[2] = m(1, 0);
    return unskewed;
}

struct se3
{
    se3() : omega(Eigen::Vector3f::Zero()), nu(Eigen::Vector3f::Zero()) { }
    se3(const Eigen::Vector3f& omega, const Eigen::Vector3f& nu) : omega(omega), nu(nu) { }
    se3(const Matrix<6, 1>& m) : omega(m.topLeftCorner<3, 1>()), nu(m.topRightCorner<3, 1>()) { }
    se3(float omega_x, float omega_y, float omega_z, float nu_x, float nu_y, float nu_z)
        : omega({omega_x, omega_y, omega_z}), nu({nu_x, nu_y, nu_z}) { }
    se3(const Eigen::Affine3f& noncanonical)
    {
        auto logged = noncanonical.matrix().log();
        omega = unskewed(logged.topLeftCorner<3, 3>());
        nu = logged.topRightCorner<3, 1>();
    }

    Eigen::Affine3f exp() const
    {
        auto skew = skewed(omega);
        auto skew2 = studd::square(skew);
        auto w_norm2 = std::max(0.000001f, omega.squaredNorm()); // because division by zero, yo
        auto w_norm = std::sqrt(w_norm2);

        Eigen::Matrix3f e_omega = Eigen::Matrix3f::Identity()
                                + (std::sin(w_norm) / w_norm) * skew
                                + ((1 - std::cos(w_norm)) / w_norm2) * skew2;
        Eigen::Matrix3f V = Eigen::Matrix3f::Identity()
                          + ((1 - std::cos(w_norm)) / w_norm2) * skew
                          + ((1 - std::sin(w_norm)) / (w_norm * w_norm2)) * skew2;

        Eigen::Matrix4f output = Eigen::Matrix4f::Zero();
        output.topLeftCorner<3, 3>() = e_omega;
        output.topRightCorner<3, 1>() = V * nu;
        output(3, 3) = 1;

        return Eigen::Affine3f(output);
    }

    friend se3 operator^(const se3& ksi, const se3& ksi_delta)
    {
        return se3(ksi.exp() * ksi_delta.exp());
    }

    friend std::ostream& operator<< (std::ostream& stream, const se3& self) {
        stream << self.omega << std::endl << self.nu;
        return stream;
    }

    Eigen::Vector3f omega;
    Eigen::Vector3f nu;
};

Eigen::Matrix3f fundamental_from_pose(const Eigen::Affine3f& pose)
{
    static Eigen::Matrix3f K_t_inv = K.transpose().inverse();

    Eigen::Vector3f e = K * pose.rotation().transpose() * pose.translation();

    return K_t_inv * pose.rotation() * skewed(e);
}

// https://forum.kde.org/viewtopic.php?f=74&t=96407
// fix to match https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#conv2d
// with multiple filters (variadic input, tuple output)
template<class Derived, class Derived2>
Derived conv2d(const Eigen::MatrixBase<Derived>& image, const Eigen::MatrixBase<Derived2>& kernel)
{
    Derived output = Derived::Zero(image.rows(), image.cols());

    typedef typename Derived::Scalar Scalar;

    Scalar normalization = kernel.sum();
    if ( normalization < 1e-6 )
    {
        normalization = 1;
    }
    for (int row = kernel.rows(); row < image.rows() - kernel.rows(); row++)
    {
        for (int col = kernel.cols(); col < image.cols() - kernel.cols(); col++)
        {
            auto block = static_cast<Derived2>(image.block(row,col,kernel.rows(),kernel.cols()));
            output.coeffRef(row,col) = block.cwiseProduct(kernel).sum();
        }
    }

    return output / normalization;
}


studd::two<Image> sobel(const Image& image)
{
    static Eigen::Matrix3f kernel_x;
    kernel_x << 1, 0, -1,
                2, 0, -2,
                1, 0, -1;

    static Eigen::Matrix3f kernel_y;
    kernel_y << 1,  2,  1,
                0,  0,  0,
               -1, -2, -1;

    return studd::make_two(conv2d(image, kernel_x), conv2d(image, kernel_y));
}

Eigen::Affine3f photometric_tracking(const std::vector<pixel>& sparse_inverse_depth,
                                     const Image& new_image,
                                     const Image& ref_image,
                                     const std::function<float(float)>& weighting)
{
    // TODO: pyramids (manual skips)

    constexpr size_t max_iterations = 5;
    constexpr float epsilon = 1e-5;

    std::cout << 1 << std::endl;

    se3 ksi;
    se3 delta_ksi;
    float last_error = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < max_iterations; i++)
    {
        se3 new_ksi;
        if (i != 0)
        {
            new_ksi = ksi ^ delta_ksi;
            std::cout << "new ksi: " << std::endl << new_ksi << std::endl;
        }
        std::cout << 2 << std::endl;

        auto warped = warp(sparse_inverse_depth, new_ksi.exp());

        // TODO: this is dumb as fuck, how can we do gradients better?
        Image warped_image = Image::Zero(height, width);
        for (size_t i = 0; i < warped.size(); i++)
        {
            int x = warped[i].x;
            int y = warped[i].y;
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                warped_image(sparse_inverse_depth[i].y, sparse_inverse_depth[i].x)
                    = new_image(y, x);
            }
        }
        std::cout << 3 << std::endl;

        studd::two<Image> gradient = sobel(warped_image);

        Eigen::VectorXf residuals;
        Eigen::DiagonalMatrix<float, Eigen::Dynamic> weights;
        std::tie(residuals, weights) = residuals_and_weights(sparse_inverse_depth, warped,
                                                             new_image, ref_image, weighting);
        std::cout << 4 << std::endl;

        float error = ((residuals.transpose() * weights).dot(residuals)
                    / sparse_inverse_depth.size());

        std::cout << "sparse_inverse_depth size = " << std::endl << sparse_inverse_depth.size() << std::endl;
        // consider switching to (last_error - error) / last_error < eps, according to wiki
        if (i > 1 && error > last_error)
        {
            std::cout << "BOOM" << std::endl;
            std::cout << "errorb: " << error << std::endl;
            //delta_ksi = transform();
            //new_ksi = transform(ksi);
            //continue;
            break;
            //exit(1);
        }
        std::cout << 4.5 << std::endl;

        Eigen::MatrixXf J = Eigen::MatrixXf(warped.size(), 6);
        for (size_t i = 0; i < sparse_inverse_depth.size(); i++)
        {
            auto& p = sparse_inverse_depth[i];
            J.row(i) = image_pose_jacobian(gradient, p.x, p.y, p.inverse_depth);
        }
        std::cout << 5 << std::endl;

        Eigen::MatrixXf J_t_weights = J.transpose() * weights;
        ksi = new_ksi;
        delta_ksi = se3((J_t_weights * J).inverse() * J_t_weights * residuals);

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

    std::cout << "returning " << std::endl << ksi << std::endl;
    std::cout << "exped " << std::endl << ksi.exp().matrix() << std::endl;
    return ksi.exp();
}

int main(int argc, const char* argv[])
{
    auto images = studd::dynamic_map<unsigned int, Image>(&load_image);

    K(0, 0) = focal_length.x();
    K(1, 1) = focal_length.y();
    K(0, 2) = principal_point.x();
    K(1, 2) = principal_point.y();
    K(2, 2) = 1;

    unsigned int frame_skip = 1;

    Eigen::Affine3f pose;
    Eigen::Matrix3f fundamental;
    std::tie(pose, fundamental) = keypoint_tracking(images[frame_skip + 1], images[1]);
    auto epipole = get_epipole(pose);
    for (size_t i = frame_skip + 1; i < 50000; i += frame_skip)
    {
        std::cout << "pose: " << std::endl << pose.matrix() << std::endl;
        std::cout << "fundamental: " << std::endl << fundamental << std::endl;
        std::cout << "m" << 0 << std::endl;
        auto epilines = generate_epilines(images[i], images[i - frame_skip], fundamental);
        std::cout << "m" << 0.1 << std::endl;
        studd::two<Image> gradient = sobel(images[i - frame_skip]);
        std::cout << "m" << 1 << std::endl;

        studd::two<Image> disparity = disparity_epiline_ssd(images[i], images[i - frame_skip],
                                                            fundamental, epilines, epipole);
        studd::two<Image> geometric = studd::make_two(Image::Zero(height, width),
                                                      Image::Zero(height, width));
        studd::two<Image> photometric = studd::make_two(Image::Zero(height, width),
                                                        Image::Zero(height, width));
        Image masked = images[i];
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
                auto photo = photometric_disparity(images[i], images[i - frame_skip], gradient,
                                                   epilines[j], x, y, disp, epipole);

                if (std::min(geo.variance, photo.variance) > 0.001)
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
                    geometric[1](y, x) = std::abs(geo.variance);
                    photometric[0](y, x) = std::abs(photo.mean);
                    photometric[1](y, x) = std::abs(photo.variance);
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
                if (std::min(geo_var, photo_var) > 0.001 || (geo_var == 0 && photo_var == 0))
                {
                    disparity[0](y, x) = -1;
                    disparity[1](y, x) = 0;
                }
                else
                {
                    // TODO: times alpha^2, figure out what alpha is
                    disparity[1](y, x) = geo_var + photo_var;
                }
            }
        }
        std::cout << "m" << 4 << std::endl;

        // 1 / Z = d / (b * f)
        float denom = ((pose.translation().norm()) * focal_length.y());
        studd::two<Image> inverse_depth = studd::make_two(disparity[0] / denom,
                                                          disparity[1] / denom);

        inverse_depth = regularize_depth(inverse_depth[0], inverse_depth[1]);
        std::cout << "m" << 5 << std::endl;

        // TODO: outlier removal

        std::vector<pixel> sparse_inverse_depth;
        sparse_inverse_depth.reserve(height * width);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                auto inv_mean = inverse_depth[0](y, x);
                auto inv_var = inverse_depth[1](y, x);
                if (inv_var >= 0 && inv_mean != 0)
                {
                    sparse_inverse_depth.emplace_back(inv_mean, inv_var, x, y);
                }
            }
        }

        pose = photometric_tracking(sparse_inverse_depth,
                                    images[i], images[i - frame_skip],
                                    [](float r) { return 1; });
        std::cout << pose.rotation() << std::endl << pose.translation() << std::endl;
        std::cout << fundamental << std::endl << epipole << std::endl;
        fundamental = fundamental_from_pose(pose);
        std::cout << fundamental << std::endl << epipole << std::endl;

        std::cout << "m" << 6 << std::endl;
        cv::waitKey(1000);
    }
}

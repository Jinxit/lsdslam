#include <iostream>
#include <string>
#include <utility>
#include <cmath>
#include <random>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include "dynamic_map.hpp"
#include "two.hpp"
#include "square.hpp"

const cv::Point2f focal_length = cv::Point2f(254.32, 375.93);
const cv::Point2f principal_point = cv::Point2f(267.38, 231.59);
constexpr size_t width = 640;
constexpr size_t height = 480;

cv::Mat load_image(const unsigned int& id)
{
    auto name = std::to_string(id);
    name.insert(0, 5 - name.size(), '0');
    return cv::imread("data/LSD_room/images/" + name + ".png", CV_LOAD_IMAGE_GRAYSCALE);
}

std::vector<cv::Point2f> fast_find(const cv::Mat& image)
{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> points;
    
    // use FAST to find features
    auto fast = cv::FastFeatureDetector::create(20);
    fast->detect(image, keypoints);
    cv::KeyPoint::convert(keypoints, points);

    // draw shit
    //cv::Mat uh;
    //cv::drawKeypoints(image, keypoints, uh, cv::Scalar(1, 0, 0));
    //cv::imshow("image", uh);
    //cv::waitKey(0);

    return points;
}

studd::two<std::vector<cv::Point2f>> klt_track_points(const cv::Mat& new_image,
                                                      const cv::Mat& ref_image)
{
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

struct transform
{
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat translation = cv::Mat::zeros(3, 1, CV_32F);

    transform() { }

    transform(const transform& rhs)
        : rotation(rhs.rotation.clone()),
          translation(rhs.translation.clone()) { }

    transform(const cv::Mat& data)
    {
        for (size_t y = 0; y < 3; y++)
        {
            for (size_t x = 0; x < 3; x++)
            {
                rotation.at<float>(y, x) = data.at<float>(y, x);
            }
            translation.at<float>(y, 0) = data.at<float>(y, 3);
        }
    }

    transform(const Eigen::Matrix4f& data)
    {
        for (size_t y = 0; y < 3; y++)
        {
            for (size_t x = 0; x < 3; x++)
            {
                rotation.at<float>(y, x) = data(y, x);
            }
            translation.at<float>(y, 0) = data(y, 3);
        }
    }

    Eigen::Matrix4f as_eigen() const
    {
        Eigen::Matrix4f output;
        for (size_t y = 0; y < 3; y++)
        {
            for (size_t x = 0; x < 3; x++)
            {
                output(y, x) = rotation.at<float>(y, x);
            }
            output(y, 3) = translation.at<float>(y, 0);
        }
        return output;
    }
};

struct epiline
{
    epiline(const cv::Point2i point, const cv::Vec3f line)
        : point(point), line(line) { };

    cv::Point2i point;
    cv::Vec3f line;
};

std::vector<epiline> generate_epilines(const cv::Mat& new_image, const cv::Mat& ref_image,
                                       const cv::Mat& fundamental)
{
    // generate points
    std::vector<cv::Point2f> points;
    points.reserve(width * height);
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            points.emplace_back(x, y);
        }
    }

    // get epilines
    // maybe just do F * p manually?
    std::vector<cv::Vec3f> lines;
    lines.reserve(width * height);
    cv::computeCorrespondEpilines(points, 1, fundamental, lines);

    // fuse them
    std::vector<epiline> epilines;
    epilines.reserve(width * height);
    for (size_t i = 0; i < lines.size(); i++)
    {
        epilines.emplace_back(points[i], lines[i]);
    }

    return epilines;
}

cv::Point2f get_epipole(const cv::Mat& fundamental)
{
    cv::Mat w;
    cv::Mat u;
    cv::Mat vt;
    cv::SVDecomp(fundamental, w, u, vt);
    cv::Mat epipole = vt(cv::Rect(0, 2, 2, 1)) / vt(cv::Rect(2, 2, 1, 1));
    return cv::Point2f(epipole);
}

studd::two<cv::Point2f> epiline_limits(cv::Vec3f epiline)
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

    return studd::make_two(cv::Point2f(x0, y0), cv::Point2f(x1, y1));
}

unsigned char interpolate(const cv::Mat& image, const cv::Point2f& p)
{
    cv::Point2i ip(p);
    cv::Point2f dp = p - cv::Point2f(ip);

    float top_left     = (1.0 - dp.x) * (1.0 - dp.y);
    float top_right    = dp.x         * (1.0 - dp.y);
    float bottom_left  = (1.0 - dp.x) * dp.y;
    float bottom_right = dp.x         * dp.y;

    return image.at<unsigned char>(ip.y,     ip.x    ) * top_left
         + image.at<unsigned char>(ip.y,     ip.x + 1) * top_right
         + image.at<unsigned char>(ip.y + 1, ip.x    ) * bottom_left
         + image.at<unsigned char>(ip.y + 1, ip.x + 1) * bottom_right;
}

cv::Mat disparity_epiline_ssd(const cv::Mat& new_image, const cv::Mat& ref_image,
                              const cv::Mat& fundamental, const std::vector<epiline>& epilines,
                              const cv::Point2f& epipole)
{
    constexpr float epiline_sample_distance = 5.0f;
    constexpr float epiline_subsample_distance = 0.1f;
    constexpr size_t num_epiline_samples = 5; // must be odd
    constexpr int half_epiline_samples = num_epiline_samples / 2;

    cv::Mat disparity = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat disps = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Rect limits(cv::Point(), disparity.size());

    auto ssd = [&](const std::array<float, num_epiline_samples> target,
                   cv::Point2f p, const cv::Point2f& dp) {
        float total = 0;
        for (size_t i = 0; i < num_epiline_samples; i++)
        {
            if (!limits.contains(p))
                total += 1000;

            float diff = target[i] - new_image.at<unsigned char>(p.y, p.x);
            total += diff * diff;
            p += dp;
        }

        return total;
    };

    auto ssd_precise = [&](cv::Point2f p, const cv::Point2f& dp) {
        float total = 0;
        for (size_t i = 0; i < num_epiline_samples; i++)
        {
            if (!limits.contains(p))
                total += 1000;

            float diff = interpolate(ref_image, p)
                       - interpolate(new_image, p);
            total += diff * diff;
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
        cv::Point2f p0, p1;
        std::tie(p0, p1) = epiline_limits(epi.line);

        std::array<float, num_epiline_samples> target{0};
        auto p = p0;
        auto dp = p1 - p0;
        dp = (dp / std::hypot(dp.x, dp.y)) * epiline_sample_distance;

        // set up target values;
        for (int i = 0; i < int(num_epiline_samples); i++)
        {
            auto point = cv::Point2f(epi.point) + dp * float(i - half_epiline_samples);

            if (limits.contains(point))
                target[i] = float(ref_image.at<unsigned char>(point.y, point.x));
        }

        // find distance for n steps
        cv::Point2f dp_total = dp * int(num_epiline_samples - 1);

        float min_ssd = std::numeric_limits<float>::infinity();
        cv::Point2f min_p(-1000, -1000);
        // keep going for the rest of the line
        while (limits.contains(p + dp_total))
        {
            if (limits.contains(p + dp_total))
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

        if (min_p.x == -1000)
        {
            continue;
        }

        /*
        p = min_p - dp;
        dp = (dp / std::hypot(dp.x, dp.y)) * epiline_subsample_distance;
        dp_total = dp * (epiline_sample_distance / epiline_subsample_distance);

        for (size_t j = 0; j < int(epiline_sample_distance / epiline_subsample_distance); j++)
        {
            if (!(limits.contains(p) && limits.contains(p + dp_total)))
            {
                continue;
            }

            float current_ssd = ssd_precise(p, dp);

            if (current_ssd < min_ssd)
            {
                min_ssd = current_ssd;
                min_p = p + dp_total / 2;
            }

            p += dp;
        }
        */

        //auto disp = std::hypot(epi.point.x - min_p.x, epi.point.y - min_p.y);
        auto dx = min_p.x - epi.point.x;
        auto dy = min_p.y - epi.point.y;
        auto disp = cv::Vec3f(std::hypot(dx, dy), dx, dy);
        //auto disp_y = -(min_p.y - epipole.y) / epi.line[0];

        //disp = min_ssd / 10;
        //disp = ref_image.at<unsigned char>(min_p.y, min_p.x);
        disparity.at<cv::Vec3f>(epi.point.y, epi.point.x) = disp;
        if (i % 1003 == 0 && false)
        {
            auto col = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
            if (dp.x < 0 && dp.y < 0)
            {
                col = cv::Scalar(255, 0, 0);
            }
            if (dp.x > 0 && dp.y < 0)
            {
                col = cv::Scalar(0, 255, 0);
            }
            if (dp.x < 0 && dp.y > 0)
            {
                col = cv::Scalar(255, 255, 0);
            }
            if (dp.x > 0 && dp.y > 0)
            {
                col = cv::Scalar(0, 0, 255);
            }
            if (dp.x == 0)
            {
                col = cv::Scalar(0, 255, 255);
            }
            if (dp.y == 0)
            {
                col = cv::Scalar(255, 0, 255);
            }
            if (dp.x == 0 && dp.y == 0)
            {
                col = cv::Scalar(255, 255, 255);
            }
            cv::line(disps, epi.point, min_p, col);
            cv::line(disps, epi.point, epi.point + cv::Point2i(dp * 25),
                     cv::Scalar(255, 255, 255));
        }

        auto disp_n = cv::norm(disp);
        if (disp_n > max_disp)
            max_disp = disp_n;
        if (disp_n < min_disp)
            min_disp = disp_n;
        avg_disp += disp_n;
    }
    avg_disp /= epilines.size();

    //std::cout << max_disp << " " << min_disp << " " << avg_disp << std::endl;
    //cv::imshow("ref", ref_image);
    //cv::imshow("new", new_image);
    //cv::imshow("disp", disparity);
    //cv::imshow("lines", disps);

    return disparity;
}

cv::Mat random_depth()
{
    cv::Mat depth = cv::Mat::zeros(height, width, CV_32F);
    static std::mt19937 gen;
    static std::uniform_real_distribution<float> dist(0, 2);

    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            depth.at<float>(y, x) = dist(gen);
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

gaussian geometric_disparity(const studd::two<cv::Mat>& gradient,
                             const epiline& epi, int x, int y,
                             const cv::Point2f& epipole)
{
    constexpr float sigma_l = 0.1f;

    cv::Point2f g = cv::Point2f(gradient[0].at<short>(y, x),
                                gradient[1].at<short>(y, x));
    if (g.x == 0 && g.y == 0)
        return gaussian(0, -1);
    g /= std::hypot(g.x, g.y);

    // semi-dense visual odometry, equation 5 and 6
    auto g_dot_l = g.x * epi.line[0] + g.y * epi.line[1];
    auto disparity = (g.x * (x - epipole.x) + g.y * (y - epipole.y)) / g_dot_l;
    auto variance = (sigma_l * sigma_l) / (g_dot_l * g_dot_l);
    return gaussian(disparity, variance);
}

gaussian photometric_disparity(const cv::Mat& new_image, const cv::Mat& ref_image,
                               const studd::two<cv::Mat>& gradient,
                               const epiline& epi, int x, int y, cv::Vec2f disparity_0,
                               const cv::Point2f& epipole)
{
    constexpr float sigma_i = 1.0f;
    cv::Point2f g = cv::Point2f(gradient[0].at<short>(y, x),
                                gradient[1].at<short>(y, x));
    auto g_p = g.x * epi.line[0] + g.y * epi.line[1];
    if (g_p == 0)
        return gaussian(0, -1);

    // semi-dense visual odometry, equation 8 and 9
    auto lambda_0 = cv::Point2f(disparity_0[0] + epi.point.x, disparity_0[1] + epi.point.y);
    auto disparity = cv::norm(disparity_0)
                   + (  ref_image.at<unsigned char>(y, x)
                      - new_image.at<unsigned char>(lambda_0.y, lambda_0.x))
                     / g_p;
    auto variance = (sigma_i * sigma_i) / (g_p * g_p);
    return gaussian(disparity, variance);
}

cv::Mat normalize_channels(const cv::Mat& input, float min, float max)
{
    std::vector<cv::Mat> channels;
    cv::split(input, channels);
    for (auto&& c : channels)
    {
        cv::normalize(c, c, min, max, cv::NORM_MINMAX);
    }
    cv::Mat output;
    cv::merge(channels, output);
    return output;
}

std::pair<transform, cv::Mat> keypoint_tracking(const cv::Mat& new_image,
                                                const cv::Mat& ref_image)
{
    auto points = klt_track_points(new_image, ref_image);

    // RANSAC
    auto fundamental = cv::findFundamentalMat(points[0], points[1], focal_length.y);
    auto essential = cv::findEssentialMat(points[0], points[1], focal_length.y,
                                          principal_point);

    transform pose;

    cv::recoverPose(essential, points[0], points[1], pose.rotation, pose.translation,
                    focal_length.y, principal_point);

    return {pose, fundamental};
}

cv::Mat regularize_depth(const cv::Mat& image)
{
    cv::Mat output = cv::Mat::zeros(height, width, CV_32FC3);

    for (int y = 1; y < int(height - 1); y++)
    {
        for (int x = 1; x < int(width - 1); x++)
        {
            int num_added = 0;

            auto& original = image.at<cv::Vec3f>(y, x);
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    auto& adjacent = image.at<cv::Vec3f>(y + dy, x + dx);
                    if (std::abs(adjacent[0] - original[0]) > original[1] * 2
                     || adjacent[1] < 0)
                        continue;

                    output.at<cv::Vec3f>(y, x)[0] += adjacent[0];
                    num_added++;
                }
            }

            if (num_added > 0)
            {
                output.at<cv::Vec3f>(y, x)[0] /= num_added;
                output.at<cv::Vec3f>(y, x)[1] = original[1];
            }
            else
            {
                output.at<cv::Vec3f>(y, x)[0] = 0;
                output.at<cv::Vec3f>(y, x)[1] = -1;
            }
        }
    }

    return output;
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
cv::Mat image_pose_jacobian(const studd::two<cv::Mat>& gradient, float x, float y, float z)
{
    cv::Mat J_I = cv::Mat::zeros(1, 2, CV_32F);
    J_I.at<float>(0, 0) = gradient[0].at<short>(y, x);
    J_I.at<float>(0, 1) = gradient[1].at<short>(y, x);

    cv::Mat J_pi = cv::Mat::zeros(2, 3, CV_32F);
    J_pi.at<float>(0, 0) =  focal_length.x / z;
    J_pi.at<float>(1, 1) =  focal_length.y / z;
    J_pi.at<float>(0, 2) = -focal_length.x * x / (z * z);
    J_pi.at<float>(1, 2) = -focal_length.y * y / (z * z);

    cv::Mat J_g = cv::Mat::zeros(3, 12, CV_32F);
    for (int i = 0; i < 3; i++)
    {
        J_g.at<float>(i, i    ) = x;
        J_g.at<float>(i, i + 3) = y;
        J_g.at<float>(i, i + 6) = z;
        J_g.at<float>(i, i + 9) = 1;
    }

    cv::Mat J_G = cv::Mat::zeros(12, 6, CV_32F);
    J_G.at<float>(1,  5) =  1;
    J_G.at<float>(2,  4) = -1;
    J_G.at<float>(3,  5) = -1;
    J_G.at<float>(5,  3) =  1;
    J_G.at<float>(6,  4) =  1;
    J_G.at<float>(7,  3) = -1;
    J_G.at<float>(9,  0) =  1;
    J_G.at<float>(10, 1) =  1;
    J_G.at<float>(11, 2) =  1;

    return J_I * J_pi * J_g * J_G;
}

std::vector<pixel> warp(std::vector<pixel> sparse_inverse_depth, const transform& ksi)
{
    for (auto&& p : sparse_inverse_depth)
    {
        // pi_1^-1
        cv::Vec3f p2;
        p2[0] = 1.0 / p.inverse_depth * (p.x + principal_point.x) / focal_length.x;
        p2[1] = 1.0 / p.inverse_depth * (p.y + principal_point.y) / focal_length.y;
        p2[2] = 1.0 / p.inverse_depth;

        // T
        p2 = cv::Mat(ksi.rotation * cv::Mat(p2) + ksi.translation);

        // pi_2
        p.x = p2[0] * focal_length.x / p2[2] - principal_point.x;
        p.y = p2[1] * focal_length.y / p2[2] - principal_point.y;
        p.inverse_depth = 1.0 / p2[2];
    }

    return sparse_inverse_depth;
}

std::pair<cv::Mat, std::vector<float>> residuals_and_weights(
    const std::vector<pixel>& sparse_inverse_depth,
    const std::vector<pixel>& warped,
    const cv::Mat& new_image, const cv::Mat& ref_image,
    const std::function<float(float)>& weighting)
{
    cv::Mat residuals = cv::Mat::zeros(sparse_inverse_depth.size(), 1, CV_32F);
    std::vector<float> weights;
    weights.reserve(sparse_inverse_depth.size());

    for (size_t i = 0; i < warped.size(); i++)
    {
        int x = warped[i].x;
        int y = warped[i].y;
        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            float ref_intensity = ref_image.at<unsigned char>(sparse_inverse_depth[i].y,
                                                              sparse_inverse_depth[i].x);
            float new_intensity = new_image.at<unsigned char>(y, x);
            auto residual = new_intensity - ref_intensity;
            residuals.at<float>(i, 0) = studd::square(residual);
            weights.emplace_back(weighting(residual));
        }
        else
        {
            weights.emplace_back(0);
        }
    }

    return {residuals, weights};
}


transform move(const transform& ksi, const transform& delta_ksi)
{
    return transform((ksi.as_eigen().exp() * delta_ksi.as_eigen().exp()).log());
}

cv::Mat multiply_diagonal(const cv::Mat& lhs_, const std::vector<float>& diag)
{
    cv::Mat lhs = lhs_.clone();

    for (int x = 0; x < lhs.cols; x++)
    {
        for (int y = 0; y < lhs.rows; y++)
        {
            if (!std::isfinite(lhs.at<float>(y, x)))
                std::cout << "! " << lhs.at<float>(y, x) << std::endl;
            if (!std::isfinite(diag[x]))
                std::cout << "!? " << diag[x] << std::endl;
            lhs.at<float>(y, x) *= diag[x];
            if (!std::isfinite(lhs.at<float>(y, x)))
                std::cout << "! " << lhs.at<float>(y, x) << std::endl;
        }
    }

    return lhs;
}

std::pair<cv::Mat, cv::Point2f> fundamental_from_pose(const transform& pose)
{
    cv::Mat K = cv::Mat::zeros(3, 3, CV_32F);
    K.at<float>(0, 0) = focal_length.x;
    K.at<float>(1, 1) = focal_length.y;
    K.at<float>(0, 2) = principal_point.x;
    K.at<float>(1, 2) = principal_point.y;
    K.at<float>(2, 2) = 1;

    std::cout << "ffp" << std::endl;
    std::cout << K << std::endl;
    cv::Mat uh = pose.rotation.t() * pose.translation;
    std::cout << uh << std::endl;
    std::cout << cv::Mat(K * pose.rotation.t() * pose.translation) << std::endl;
    cv::Vec3f e = cv::Mat(K * pose.rotation.t() * pose.translation);
    std::cout << e << std::endl;

    cv::Mat e_skew = cv::Mat::zeros(3, 3, CV_32F);
    e_skew.at<float>(1, 0) =  e[2];
    e_skew.at<float>(0, 1) = -e[2];
    e_skew.at<float>(0, 2) =  e[1];
    e_skew.at<float>(2, 0) = -e[1];
    e_skew.at<float>(2, 1) =  e[0];
    e_skew.at<float>(1, 2) = -e[0];
    std::cout << e_skew << std::endl;

    return {K.t().inv() * pose.rotation * e_skew, cv::Point2f(e[0], e[1])};
}

transform photometric_tracking(const std::vector<pixel>& sparse_inverse_depth,
                               const cv::Mat& new_image, const cv::Mat& ref_image,
                               const std::function<float(float)>& weighting)
{
    // TODO: pyramids
    // TODO: pyrDown does gaussian convolution, maybe skip?
    //cv::Mat new_image2;
    //cv::pyrDown(new_image, new_image2);

    constexpr size_t max_iterations = 5;
    constexpr float epsilon = 1e-5;

    std::cout << 1 << std::endl;

    transform ksi;
    transform delta_ksi;
    float last_error = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < max_iterations; i++)
    {
        transform new_ksi;
        if (i != 0)
        {
            new_ksi = move(ksi, delta_ksi);
        }
        std::cout << 2 << std::endl;

        auto warped = warp(sparse_inverse_depth, new_ksi);

        // TODO: this is dumb as fuck, how can we do gradients better?
        cv::Mat warped_image = cv::Mat::zeros(height, width, CV_8U);
        for (size_t i = 0; i < warped.size(); i++)
        {
            int x = warped[i].x;
            int y = warped[i].y;
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                warped_image.at<unsigned char>(sparse_inverse_depth[i].y,
                                               sparse_inverse_depth[i].x)
                    = new_image.at<unsigned char>(y, x);
            }
        }
        std::cout << 3 << std::endl;

        studd::two<cv::Mat> gradient;
        cv::Sobel(warped_image, gradient[0], CV_16S, 1, 0);
        cv::Sobel(warped_image, gradient[1], CV_16S, 0, 1);

        cv::Mat residuals;
        std::vector<float> weights;
        std::tie(residuals, weights) = residuals_and_weights(sparse_inverse_depth, warped,
                                                             new_image, ref_image, weighting);
        std::cout << 4 << std::endl;

        cv::Mat residuals_t_weights = multiply_diagonal(cv::Mat(residuals.t()), weights);
        std::cout << 4.1 << std::endl;

        float error = cv::Mat((residuals_t_weights * residuals)
                             / sparse_inverse_depth.size()).at<float>(0, 0);

        // consider switching to (last_error - error) / last_error < eps, according to wiki
        if (error > last_error)
        {
            std::cout << "BOOM" << std::endl;
            delta_ksi = transform();
            new_ksi = transform(ksi);
            //continue;
            //exit(1);
        }
        std::cout << 4.5 << std::endl;

        cv::Mat J = cv::Mat::zeros(warped.size(), 6, CV_32F);
        for (size_t i = 0; i < sparse_inverse_depth.size(); i++)
        {
            auto& p = sparse_inverse_depth[i];
            // what the fuck openCV?
            cv::Mat temp = image_pose_jacobian(gradient, p.x, p.y, 1.0 / p.inverse_depth);
            temp.row(0).copyTo(J.row(i));
        }
        std::cout << 5 << std::endl;

        cv::Mat J_t_weights = multiply_diagonal(cv::Mat(J.t()), weights);
        ksi = transform(new_ksi);
        delta_ksi = transform(cv::Mat((J_t_weights * J).inv() * J_t_weights * residuals));

        std::cout << 6 << std::endl;
        std::cout << "ksi: " << std::endl << ksi.rotation << std::endl
                  << ksi.translation << std::endl;
        std::cout << "delta_ksi: " << std::endl << delta_ksi.rotation << std::endl
                  << delta_ksi.translation << std::endl;
        std::cout << "error: " << error << std::endl;
        std::cout << "first: " << cv::Mat((J_t_weights * J).inv()) << std::endl;
        if (std::abs(error - last_error) < epsilon)
            break;

        last_error = error;
    }

    return ksi;
}

int main(int argc, const char* argv[])
{
    auto images = studd::dynamic_map<unsigned int, cv::Mat>(&load_image);

    int frame_skip = 1;

    transform pose;
    cv::Mat fundamental;
    std::tie(pose, fundamental) = keypoint_tracking(images[frame_skip + 1], images[1]);
    auto epipole = get_epipole(fundamental);
    for (size_t i = frame_skip + 1; i < 50000; i += frame_skip)
    {
        std::cout << "m" << 0 << std::endl;
        auto epilines = generate_epilines(images[i], images[i - frame_skip], fundamental);
        std::cout << "m" << 0.1 << std::endl;
        cv::imshow("original", images[i]);
        studd::two<cv::Mat> gradient;
        std::cout << "m" << 0.2 << std::endl;
        cv::Sobel(images[i - frame_skip], gradient[0], CV_16S, 1, 0);
        cv::Sobel(images[i - frame_skip], gradient[1], CV_16S, 0, 1);

        std::cout << "m" << 1 << std::endl;

        auto disparity = disparity_epiline_ssd(images[i], images[i - frame_skip],
                                               fundamental, epilines, epipole);
        cv::Mat geometric = cv::Mat::zeros(height, width, CV_32FC3);
        cv::Mat photometric = cv::Mat::zeros(height, width, CV_32FC3);
        cv::Mat masked = images[i].clone();
        std::cout << "m" << 2 << std::endl;
        size_t j = 0;
        size_t geo_wins = 0;
        size_t photo_wins = 0;
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                auto disp = disparity.at<float>(y, x);
                auto geo = geometric_disparity(gradient, epilines[j], x, y, epipole);
                auto photo = photometric_disparity(images[i], images[i - frame_skip], gradient,
                                                   epilines[j], x, y, disp, epipole);

                if (std::min(geo.variance, photo.variance) > 0.001)
                {
                    masked.at<unsigned char>(y, x) = 0;
                    if (photo.variance < geo.variance)
                        photo_wins++;
                    else
                        geo_wins++;
                    geometric.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
                    photometric.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
                }
                else
                {
                    geometric.at<cv::Vec3f>(y, x) = cv::Vec3f(std::abs(geo.mean),
                                                              geo.variance,
                                                              0);
                    photometric.at<cv::Vec3f>(y, x) = cv::Vec3f(std::abs(photo.mean),
                                                                photo.variance,
                                                                0);
                }

                disparity.at<cv::Vec3f>(y, x)[1] = 0;
                disparity.at<cv::Vec3f>(y, x)[2] = 0;
                j++;
            }
        }

        std::cout << "m" << 3 << std::endl;
        // filter out bad candidates
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                auto geo_var = geometric.at<cv::Vec3f>(y, x)[1];
                auto photo_var = photometric.at<cv::Vec3f>(y, x)[1];
                if (std::min(geo_var, photo_var) > 0.001 || (geo_var == 0 && photo_var == 0))
                {
                    disparity.at<cv::Vec3f>(y, x) = cv::Vec3f(-1, 0, 0);
                }
                else
                {
                    // TODO: times alpha^2, figure out what alpha is
                    disparity.at<cv::Vec3f>(y, x)[1] = geo_var + photo_var;
                }
            }
        }
        std::cout << "m" << 4 << std::endl;

        // 1 / Z = d / (b * f)
        cv::Mat inverse_depth = disparity / (cv::norm(pose.translation) * focal_length.y);

        inverse_depth = regularize_depth(inverse_depth);
        std::cout << "m" << 5 << std::endl;

        // TODO: outlier removal

        std::vector<pixel> sparse_inverse_depth;
        sparse_inverse_depth.reserve(height * width);
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                auto& inv = inverse_depth.at<cv::Vec3f>(y, x);
                if (inv[1] >= 0 && inv[0] != 0)
                {
                    sparse_inverse_depth.emplace_back(inv[0], inv[1], x, y);
                }
            }
        }

        pose = photometric_tracking(sparse_inverse_depth,
                                    images[i], images[i - frame_skip],
                                    [](float r) { return 1; });
        std::cout << pose.rotation << std::endl << pose.translation << std::endl;
        std::cout << fundamental << std::endl << epipole << std::endl;
        std::tie(fundamental, epipole) = fundamental_from_pose(pose);
        std::cout << fundamental << std::endl << epipole << std::endl;

        std::cout << "m" << 6 << std::endl;
        //inverse_depth = normalize_channels(inverse_depth, 0, 4);
        geometric = normalize_channels(geometric, 0, 4);
        photometric = normalize_channels(photometric, 0, 4);
        //cv::sqrt(inverse_depth, inverse_depth);
        cv::sqrt(geometric, geometric);
        cv::sqrt(photometric, photometric);

        for (size_t j = 0; j < 10; j++)
        {
            cv::line(geometric, epilines[j * epilines.size() / 11].point, epipole,
                     cv::Scalar(0, 0, 0.8));
            cv::line(photometric, epilines[j * epilines.size() / 11].point, epipole,
                     cv::Scalar(0, 0, 0.8));
        }

        cv::imshow("geometric", geometric);
        cv::imshow("photometric", photometric);
        cv::imshow("masked", masked);
        std::cout << "photo wins: " << photo_wins << std::endl;
        std::cout << "geo wins: " << geo_wins << std::endl;
        std::cout << "ratio " << (float(photo_wins) / std::min(photo_wins, geo_wins))
                  << ":" << (float(geo_wins) / std::min(photo_wins, geo_wins)) << std::endl;
        //cv::imshow("gradients", gradient[0] + gradient[1]);
        cv::imshow("disparity", inverse_depth);
        //cv::imshow("depth", depth);

        // TODO: replace with depth tracking
        //std::tie(pose, fundamental) = keypoint_tracking(images[i], images[i - frame_skip]);
        
        std::cout << "m" << 7 << std::endl;
        cv::waitKey(1000);
    }
}

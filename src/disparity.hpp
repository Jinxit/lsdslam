#pragma once

#include <array>

#include <eigen3/Eigen/Dense>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "two.hpp"
#include "epiline.hpp"
#include "eigen_utils.hpp"
#include "gaussian.hpp"
#include "misc_utils.hpp"

inline gaussian geometric_disparity(const studd::two<Image>& gradient,
                                    const epiline& epi, int x, int y,
                                    const Eigen::Vector2f& epipole)
{
    constexpr float sigma_l = 0.05f;

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

inline gaussian photometric_disparity(const Image& new_image,
                                      const Image& ref_image,
                                      const studd::two<Image>& gradient,
                                      const epiline& epi, int x, int y,
                                      const Eigen::Vector2f& disparity_0,
                                      const Eigen::Vector2f& epipole)
{
    constexpr float sigma_i = 0.05f;

    Eigen::Vector2f g(gradient[0](y, x), gradient[1](y, x));
    auto g_p = g.x() * epi.line[0] + g.y() * epi.line[1];
    if (g_p == 0)
        return gaussian(0, -1);

    // semi-dense visual odometry, equation 8 and 9
    Eigen::Vector2f lambda_0 = Eigen::Vector2f(disparity_0[0] + epi.point.x(),
                                               disparity_0[1] + epi.point.y());
    float lambda_disp = 0;
    if (lambda_0.x() >= 0 && lambda_0.x() < new_image.cols() &&
        lambda_0.y() >= 0 && lambda_0.y() < new_image.rows())
    {
        lambda_disp = new_image(lambda_0.y(), lambda_0.x());
    }
    auto disparity = disparity_0.norm() + (ref_image(y, x) - lambda_disp) / g_p;
    auto variance = 2 * (sigma_i * sigma_i) / (g_p * g_p);
    return gaussian(disparity, variance);
}

inline studd::two<Eigen::Vector2f> epiline_limits(Eigen::Vector3f epiline,
                                                  int height, int width)
{
    auto A = epiline[0];
    auto B = epiline[1];
    auto C = epiline[2];

    if ((A == 0 && B == 0) || A != A || B != B)
        return {Eigen::Vector2f(0, 0), Eigen::Vector2f(0, 0)};

    auto a = Eigen::Vector2f(0, -C / B);
    auto b = Eigen::Vector2f((-B * (height - 1) - C) / A, height - 1);
    auto c = Eigen::Vector2f(width - 1, (-A * (width - 1) - C) / B);
    auto d = Eigen::Vector2f(-C / A, 0);

    auto safe = [&](const Eigen::Vector2f& p) {
        return p.x() >= 0 && p.y() >= 0 && int(p.x()) < width && int(p.y()) < height;
    };

    for (auto&& p1 : {a, b, c, d})
    {
        for (auto&& p2 : {a, b, c, d})
        {
            if (p1 == p2)
                continue;

            if (safe(p1) && safe(p2))
            {
                //std::cout << "before: " << A << ", " << B << ", " << C << std::endl;
                //std::cout << "after: " << p1.x() << ", " << p1.y() << "; " << p2.x() << ", " << p2.y() << std::endl;
                if (p1.x() < p2.x())
                    return {p2, p1};
                else
                    return {p1, p2};
            }
        }
    }
    return {Eigen::Vector2f(0, 0), Eigen::Vector2f(0, 0)};
}

inline studd::two<Eigen::Vector2f> epiline_limits2(Eigen::Vector3f epiline,
                                                  int height, int width)
{
    std::cout << "before: " << epiline[0] << ", " << epiline[1] << ", " << epiline[2] << std::endl;
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
    std::cout << "after: " << x0 << ", " << y0 << "; " << x1 << ", " << y1 << std::endl;

    return studd::make_two(Eigen::Vector2f(x0, y0), Eigen::Vector2f(x1, y1));
}

inline studd::two<Image> disparity_epilines(const Image& new_image, const Image& ref_image,
                                            const std::vector<epiline>& epilines,
                                            const Eigen::Vector2f& epipole)
{
    constexpr float epiline_sample_distance = 1.0f;
    constexpr size_t num_epiline_samples = 5; // must be odd
    constexpr int half_epiline_samples = num_epiline_samples / 2;

    auto height = new_image.rows();
    auto width = new_image.cols();

    studd::two<Image> disparity = studd::make_two(Image::Zero(height, width),
                                                  Image::Zero(height, width));

    auto safe = [&](const Eigen::Vector2f& p) {
        return p.x() >= 0 && p.y() >= 0 && p.x() < width - 1 && p.y() < height - 1;
    };

    auto ssd = [&](const std::array<float, num_epiline_samples> target,
                   Eigen::Vector2f p, const Eigen::Vector2f& dp)
    {
        float total = 0;
        for (size_t i = 0; i < num_epiline_samples; i++)
        {
            total += studd::square(target[i] - interpolate(new_image, p));

            p += dp;
        }

        return total;
    };

    for (auto&& epi : epilines)
    {
        Eigen::Vector2f p0, p1;
        std::tie(p0, p1) = epiline_limits(epi.line, height, width);

        std::array<float, num_epiline_samples> target{{0}};
        Eigen::Vector2f p = p0;
        Eigen::Vector2f dp = (p1 - p0).normalized() * epiline_sample_distance;

        if (p0 == p1)
        {
            disparity[0](epi.point.y(), epi.point.x()) =
            disparity[1](epi.point.y(), epi.point.x()) =
                std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        // set up target values;
        for (int i = 0; i < int(num_epiline_samples); i++)
        {
            Eigen::Vector2f point = epi.point.cast<float>()
                                  + dp * float(i - half_epiline_samples);

            if (safe(point))
                target[i] = interpolate(ref_image, point);
        }

        // find distance for n steps
        Eigen::Vector2f dp_total = dp * (num_epiline_samples - 1);

        float min_ssd = std::numeric_limits<float>::infinity();
        Eigen::Vector2f min_p(-1000, -1000);
        // keep going for the rest of the line
        while (safe(p + dp_total))
        {
            if (safe(p) && safe(p + dp_total))
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
            disparity[0](epi.point.y(), epi.point.x()) =
            disparity[1](epi.point.y(), epi.point.x()) =
                std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        disparity[0](epi.point.y(), epi.point.x()) = min_p.x() - epi.point.x();
        disparity[1](epi.point.y(), epi.point.x()) = min_p.y() - epi.point.y();
    }
    show("x_disp", disparity[0], true);
    show("y_disp", disparity[1], true);
    show_disparity("l_disp", disparity, ref_image, true);

    return disparity;
}

inline studd::two<Image> disparity_rectified(const Image& new_image, const Image& ref_image,
                                             const studd::two<Image>& gradient)
{
    auto block_size = 11;
    auto sbm = cv::StereoSGBM::create(0, 256, block_size, 8 * block_size, 32 * block_size, 30, 0,
                                      10, 100, 2, cv::StereoSGBM::MODE_HH);
    //auto sbm = cv::StereoBM::create(96, 21);

    cv::Mat leftimage, rightimage, disp;
    cv::eigen2cv(new_image, rightimage);
    rightimage.convertTo(rightimage, CV_8U, 255);
    cv::eigen2cv(ref_image, leftimage);
    leftimage.convertTo(leftimage, CV_8U, 255);
    sbm->compute(rightimage, leftimage, disp);
    //cv::imshow("left", leftimage);
    //cv::imshow("right", rightimage);
    Image uh;
    disp = disp / 16;
    cv::cv2eigen(disp, uh);
    disp.convertTo(disp, CV_8U);
    //cv::imshow("dispcv", disp);
    //show_rainbow("dispeig", studd::two<Image>(uh, Image::Constant(new_image.rows(),
    //                                                              new_image.cols(), 1).eval()),
    //             new_image);
    //cv::waitKey(0);




    // TODO: split into sample distance and sample delta
    constexpr float epiline_sample_distance = 1.0f;
    constexpr float gradient_epsilon = 4e-1;
    constexpr size_t num_epiline_samples = 7; // must be odd
    constexpr int disparity_range = 50;

    int height = new_image.rows();
    int width = new_image.cols();

    studd::two<Image> disparity = studd::make_two(Image::Zero(height, width),
                                                  Image::Zero(height, width));

    auto ssd = [&](const std::array<float, num_epiline_samples> target,
                   float x, float y, float dx)
    {
        float total = 0;
        for (size_t i = 0; i < num_epiline_samples; i++)
        {
            if (!(x >= 0 && x < width - 1))
                total += studd::square(target[i]);
            else
                total += studd::square(target[i] - interpolate(new_image,
                                                               Eigen::Vector2f(x, y)));

            x += dx;
        }

        return total;
    };

    constexpr int half_epiline_samples = num_epiline_samples / 2;
    Eigen::Vector2f dp(epiline_sample_distance, 0);
    for (int yo = 0; yo < height - 1; yo++)
    {
        for (int xo = 0; xo < width - 1; xo++)
        {
            if (std::abs(gradient[0](yo, xo)) < gradient_epsilon)
            {
                disparity[1](yo, xo) = -1;
                continue;
            }
            if (uh(yo, xo) >= 0)
            {
                disparity[0](yo, xo) = uh(yo, xo);
                disparity[1](yo, xo) = 1;
                continue;
            }

            // set up target values;
            std::array<float, num_epiline_samples> target{{0}};
            for (int i = 0; i < int(num_epiline_samples); i++)
            {
                Eigen::Vector2f point = Eigen::Vector2f(xo + epiline_sample_distance
                                                           * float(i - half_epiline_samples),
                                                        yo);

                if (point.x() >= 0 && point.x() < width - 1)
                    target[i] = interpolate(ref_image, point);
            }

            float min_ssd = std::numeric_limits<float>::infinity();
            float min_x = -1000;
            for (float x0 = xo - disparity_range; x0 < xo + disparity_range;
                 x0 += epiline_sample_distance)
            {
                auto sx = x0 - epiline_sample_distance * half_epiline_samples;
                float current_ssd = ssd(target, sx, yo, epiline_sample_distance);

                if (current_ssd < min_ssd)
                {
                    min_ssd = current_ssd;
                    min_x = x0;
                }
            }

            if (min_x == -1000)
            {
                disparity[1](yo, xo) = -1;
            }
            else
            {
                disparity[0](yo, xo) = min_x - xo;
            }
        }
    }

    return disparity;
}
#pragma once

#include <iostream>
#include <unordered_map>

#include <eigen3/Eigen/Dense>

#include "two.hpp"
#include "square.hpp"
#include "gaussian.hpp"
#include "vector_map.hpp"
#include "matrix_hash.hpp"

template<int Height, int Width>
using Matrix = Eigen::Matrix<float, Height, Width>;
using Image = Eigen::MatrixXf;
using Imaged = Eigen::MatrixXd;
using sparse_gaussian = vector_map<Eigen::Vector2i, gaussian,
                                   matrix_hash<Eigen::Vector2i>>;

template<class T>
inline Eigen::Matrix<T, 3, 3> skewed(const Eigen::Matrix<T, 3, 1>& x)
{
    Eigen::Matrix<T, 3, 3> skewed;
    skewed << T(0), -x[2],  x[1],
              x[2],  T(0), -x[0],
             -x[1],  x[0],  T(0);
    return skewed;
}

template<class T>
inline Eigen::Matrix<T, 3, 1> unskewed(const Eigen::Matrix<T, 3, 3>& m)
{
    Eigen::Matrix<T, 3, 1> unskewed;
    unskewed[0] = m(2, 1);
    unskewed[1] = m(0, 2);
    unskewed[2] = m(1, 0);
    return unskewed;
}

// https://forum.kde.org/viewtopic.php?f=74&t=96407
template<class ImageMatrix, class KernelMatrix>
ImageMatrix conv2d(const Eigen::MatrixBase<ImageMatrix>& image, const Eigen::Vector2i& stride,
                   const Eigen::MatrixBase<KernelMatrix>& kernel)
{
    ImageMatrix output(image.rows(), image.cols());
    int half_rows = kernel.rows() / 2;
    int half_cols = kernel.cols() / 2;

    for (int row = 0; row < image.rows(); row += stride.y())
    {
        for (int col = 0; col < image.cols(); col += stride.x())
        {
            int a, b, c, d, y, x;

            y = std::max(0, row - half_rows);
            x = std::max(0, col - half_cols);
            a = std::max(0, std::min(half_rows, half_rows - row));
            b = std::max(0, std::min(half_cols, half_cols - col));
            c = std::min(kernel.rows() - a, image.rows() - row + 1);
            d = std::min(kernel.cols() - b, image.cols() - col + 1);

            output(row, col) = image.block(y, x, c, d)
                               .cwiseProduct(kernel.block(a, b, c, d)).sum();
        }
    }

    return output;
}

inline studd::two<Image> sobel(const Image& image)
{
    static Eigen::Matrix3f kernel_x;
    kernel_x << -1, 0, 1,
                -2, 0, 2,
                -1, 0, 1;

    static Eigen::Matrix3f kernel_y;
    kernel_y << -1,  -2,  -1,
                 0,   0,   0,
                 1,   2,   1;

    return {conv2d(image, Eigen::Vector2i::Ones(), kernel_x),
            conv2d(image, Eigen::Vector2i::Ones(), kernel_y)};
}

template<class ImageMatrix>
ImageMatrix max_pool(const Eigen::MatrixBase<ImageMatrix>& image, const Eigen::Vector2i& stride)
{
    int sx = stride.x();
    int sy = stride.y();
    ImageMatrix output(image.rows() / sy, image.cols() / sx);

    for (int row = 0; row < output.rows(); row++)
    {
        for (int col = 0; col < output.cols(); col++)
        {
            output(row, col) = image.block(row * sy, col * sx, sy, sx).maxCoeff();
        }
    }

    return output;
}

inline Eigen::Affine3f interpolate(const Eigen::Affine3f lhs, const Eigen::Affine3f rhs,float t)
{
    Eigen::Affine3f result;
    result.matrix().topLeftCorner<3, 3>() = Eigen::Quaternionf(lhs.rotation().matrix())
                                            .slerp(t, Eigen::Quaternionf(rhs.rotation()))
                                            .matrix();
    result.translation() = (1 - t) * lhs.translation() + t * rhs.translation();
    return result;
}

inline float interpolate(const Image& image, const Eigen::Vector2f& p)
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

inline studd::two<Image> densify_depth(const sparse_gaussian& sparse_inverse_depth,
                                       int height, int width, int pyramid_level = 1)
{
    Image inv_depth = Image::Zero(height, width);
    Image variance = Image::Constant(height, width, -1);

    for (const auto& kvp : sparse_inverse_depth)
    {
        size_t x = kvp.first.x() / pyramid_level;
        size_t y = kvp.first.y() / pyramid_level;

        if (x >= 0 && x < width && y >= 0 && y < height &&
            inv_depth(y, x) < kvp.second.mean)
        {
            inv_depth(y, x) = kvp.second.mean;
            variance(y, x) = kvp.second.variance;
        }
    }

    return {inv_depth, variance};
}

inline sparse_gaussian sparsify_depth(const studd::two<Image>& inverse_depth, bool invert_depth = false)
{
    auto height = inverse_depth[0].rows();
    auto width = inverse_depth[0].cols();

    sparse_gaussian sparse_inverse_depth;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            auto inv_mean = inverse_depth[0](y, x);
            auto inv_var = inverse_depth[1](y, x);

            if (inv_var >= 0 && inv_mean != 0)
            {
                sparse_inverse_depth.push_back(
                    {x, y}, gaussian(invert_depth ? 1.0 / inv_mean : inv_mean, inv_var)
                );
            }
        }
    }

    return sparse_inverse_depth;
}

inline sparse_gaussian warp(sparse_gaussian sparse_inverse_depth,
                            const Eigen::Matrix3f& intrinsic,
                            const Eigen::Affine3f& ksi)
{
    double f_x = intrinsic(0, 0);
    double f_y = intrinsic(1, 1);
    double c_x = intrinsic(0, 2);
    double c_y = intrinsic(1, 2);

    for (auto& kvp : sparse_inverse_depth)
    {
        double x = kvp.first.x();
        double y = kvp.first.y();
        double inv_depth = kvp.second.mean;
        double variance = kvp.second.variance;

        // pi_1^-1
        Eigen::Vector3f p2(
            (x - c_x) / (f_x * inv_depth),
            (y - c_y) / (f_y * inv_depth),
            1.0 / inv_depth
        );

        // T

        p2 = ksi * p2;
        if (p2[2] > 0)
        {
            // pi_2
            kvp.first.x() = std::round(p2.x() * (f_x / p2.z()) + c_x);
            kvp.first.y() = std::round(p2.y() * (f_y / p2.z()) + c_y);
            kvp.second.mean = 1.0 / p2.z();
            kvp.second.variance = std::pow(inv_depth / kvp.second.mean, 4) * variance;
        }
        else
        {
            kvp.second.mean = 0;
            kvp.second.variance = -1;
        }
    }

    return sparse_inverse_depth;
}

inline Image pyramid(const Image& src, int pyramid)
{
    size_t height = src.rows();
    size_t width = src.cols();

    Image resized = Image::Zero(height / pyramid + 1, width / pyramid + 1);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            resized(y / pyramid, x / pyramid) += src(y, x) / studd::square(pyramid);
        }
    }

    return resized;
}
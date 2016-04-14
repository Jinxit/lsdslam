#pragma once

#include <eigen3/Eigen/Dense>

#include "two.hpp"
#include "square.hpp"

template<int Height, int Width>
using Matrix = Eigen::Matrix<float, Height, Width>;
using Image = Eigen::MatrixXf;

inline Eigen::Matrix3f skewed(const Eigen::Vector3f& x)
{
    Eigen::Matrix3f skewed;
    skewed <<    0, -x[2],  x[1],
              x[2],     0, -x[0],
             -x[1],  x[0],     0;
    return skewed;
}

inline Eigen::Vector3f unskewed(const Eigen::Matrix3f& m)
{
    Eigen::Vector3f unskewed;
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

    return studd::make_two(conv2d(image, Eigen::Vector2i::Ones(), kernel_x),
                           conv2d(image, Eigen::Vector2i::Ones(), kernel_y));
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

inline Eigen::Affine3f interpolate(const Eigen::Affine3f lhs, const Eigen::Affine3f rhs, float t)
{
    Eigen::Affine3f result;
    result.matrix().topLeftCorner<3, 3>() = Eigen::Quaternionf(lhs.rotation().matrix())
                                            .slerp(t, Eigen::Quaternionf(rhs.rotation()))
                                            .matrix();
    result.translation() = (1 - t) * lhs.translation() + t * rhs.translation();
    return result;
}
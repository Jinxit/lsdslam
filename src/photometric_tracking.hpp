#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

#include <sophus/se3.hpp>

#include "local_parameterization.hpp"
#include "two.hpp"
#include "square.hpp"
#include "eigen_utils.hpp"
#include "misc_utils.hpp"
#include "static_cast_func.hpp"

using Grid = ceres::Grid2D<float, 1, false, false>;

template<template<class, int> class G>
struct warp_error
{
    warp_error(const Eigen::Matrix3f& intrinsic, const Grid* const new_grid,
               const Image* const ref_image, const Image* const ref_variance,
               float x, float y, float inv_depth, float variance)
        : new_grid(new_grid), new_interp(*new_grid),
          ref_image(ref_image), ref_variance(ref_variance),
          f_x(intrinsic(0, 0)), f_y(intrinsic(1, 1)),
          c_x(intrinsic(0, 2)), c_y(intrinsic(1, 2)),
          x(x), y(y), inv_depth(inv_depth), variance(variance) { }

    template<class T>
    bool operator()(const T* const transform_, T* residuals) const
    {
        T variance = T((*ref_variance)(int(y), int(x)));
        if (variance <= T(0))
        {
            residuals[0] = T(0);
            return true;
        }

        G<T, 0> transform;
        std::copy(transform_, transform_ + G<T, 0>::num_parameters, transform.data());

        // TODO: replace with project/unproject functions
        Eigen::Matrix<T, 3, 1> p2 = transform * Eigen::Matrix<T, 3, 1>(
            T((x - c_x + 0.5f) / (f_x * inv_depth)),
            T((y - c_y + 0.5f) / (f_y * inv_depth)),
            T(1.0 / inv_depth)
        );

        T x2 = p2.x() * (T(f_x) / p2.z()) + T(c_x);
        T y2 = p2.y() * (T(f_y) / p2.z()) + T(c_y);

        //Grid new_depth_grid(new_depth->data(), 0, new_depth->rows(), 0, new_depth->cols());
        //ceres::BiCubicInterpolator<Grid> new_depth_interp(new_depth_grid);

        T new_intensity;
        new_interp.Evaluate(y2, x2, &new_intensity);
        T ref_intensity = T((*ref_image)(int(y), int(x)));
        //T new_depth_pixel;
        //new_depth_interp.Evaluate(y2, x2, &new_depth_pixel);
        //TODO why does tracking suddenly suck?
        residuals[0] = T(ref_intensity - new_intensity) / variance;
        //residuals[1] = T(1.0 / inv_depth) - new_depth_pixel;
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Matrix3f& intrinsic,
                                       const Grid* const new_grid,
                                       const Image* const ref_image,
                                       const Image* const ref_variance,
                                       float x, float y, float inv_depth, float variance)
    {
        return (new ceres::AutoDiffCostFunction<warp_error, 1, G<double, 0>::num_parameters>(
                    new warp_error(intrinsic, new_grid, ref_image, ref_variance, x, y,
                                   inv_depth, variance)));
    }

    //const Image* const new_image;
    const Grid* const new_grid;
    ceres::BiCubicInterpolator<Grid> new_interp;
    const Image* const ref_image;
    const Image* const ref_variance;
    float f_x;
    float f_y;
    float c_x;
    float c_y;
    float x;
    float y;
    float inv_depth;
    float variance;
};

template<template<class, int> class G, class T, int O, class LossFunc>
G<T, O> photometric_tracking(const sparse_gaussian& sparse_inverse_depth,
                          const Image& new_image, const Image& ref_image,
                          const Image& ref_variance,
                          const Eigen::Matrix3f& intrinsic,
                          const LossFunc& loss_func,
                          const ceres::Solver::Options options,
                          const G<T, O>& guess,
                          int max_pyramid)
{
    auto height = new_image.rows();
    auto width = new_image.cols();

    G<double, O> ksi = guess.template cast<double>();

    int pyr_i = -1;
    for (int pyr = max_pyramid; pyr > 0; pyr /= 2)
    {
        pyr_i++;

        Eigen::Matrix3f pyramid_intrinsic = intrinsic / pyr;
        pyramid_intrinsic(2, 2) = 1;

        auto ref_resized = pyramid(ref_image, pyr);
        auto new_resized = pyramid(new_image, pyr);
        Grid new_grid(new_resized.data(), 0, new_resized.rows(), 0, new_image.cols());

        ceres::Problem problem;
        problem.AddParameterBlock(ksi.data(), G<T, O>::num_parameters,
                                  new Sophus::LocalParameterization<G<double, O>, true>);

        for (size_t i = 0; i < sparse_inverse_depth.size(); i += 1)
        {
            auto cost_func = warp_error<G>::create(
                pyramid_intrinsic,
                &new_grid,
                &ref_resized,
                &ref_variance,
                sparse_inverse_depth[i].first.x() / pyr,
                sparse_inverse_depth[i].first.y() / pyr,
                sparse_inverse_depth[i].second.mean,
                sparse_inverse_depth[i].second.variance);
            problem.AddResidualBlock(cost_func, new ceres::HuberLoss(loss_func), ksi.data());
        }

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.FullReport() << std::endl;
    }

    G<T, O> ksi_ = ksi.template cast<T>();
    return ksi_;
}

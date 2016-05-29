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

template<template<class, int> class G>
struct warp_error
{
    warp_error(const Eigen::Matrix3f& intrinsic, const Image* const new_image,
               const Image* const ref_image, float x, float y, float inv_depth)
        : new_image(new_image), ref_image(ref_image),
          f_x(intrinsic(0, 0)), f_y(intrinsic(1, 1)),
          c_x(intrinsic(0, 2)), c_y(intrinsic(1, 2)),
          x(x), y(y), inv_depth(inv_depth) { }

    template<class T>
    bool operator()(const T* const transform_, T* residuals) const
    {
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

        // TODO make member
        using Grid = ceres::Grid2D<float, 1, false, false>;
        Grid new_grid(new_image->data(), 0, new_image->rows(), 0, new_image->cols());
        ceres::BiCubicInterpolator<Grid> new_interp(new_grid);

        T new_intensity;
        new_interp.Evaluate(y2, x2, &new_intensity);
        T ref_intensity = T((*ref_image)(int(y), int(x)));
        residuals[0] = T(ref_intensity - new_intensity);
        // TODO: variance
        // kvp.second.variance = std::pow(inv_depth / kvp.second.mean, 4) * variance;
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Matrix3f& intrinsic,
                                       const Image* const new_image,
                                       const Image* const ref_image,
                                       float x, float y, float inv_depth)
    {
        return (new ceres::AutoDiffCostFunction<warp_error, 1, G<double, 0>::num_parameters>(
                    new warp_error(intrinsic, new_image, ref_image, x, y, inv_depth)));
    }

    const Image* const new_image;
    const Image* const ref_image;
    float f_x;
    float f_y;
    float c_x;
    float c_y;
    float x;
    float y;
    float inv_depth;
};

template<template<class, int> class G, class T, int O>
G<T, O> photometric_tracking(const sparse_gaussian& sparse_inverse_depth,
                          const Image& new_image, const Image& ref_image,
                          const Eigen::Matrix3f& intrinsic,
                          const std::function<float(float)>& weighting,
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

        ceres::Problem problem;
        problem.AddParameterBlock(ksi.data(), G<T, O>::num_parameters,
                                  new Sophus::LocalParameterization<G<double, O>, true>);

        for (size_t i = 0; i < sparse_inverse_depth.size(); i += pyr)
        {
            auto cost_func = warp_error<G>::create(
                pyramid_intrinsic,
                &new_resized,
                &ref_resized,
                sparse_inverse_depth[i].first.x() / pyr,
                sparse_inverse_depth[i].first.y() / pyr,
                sparse_inverse_depth[i].second.mean);
            problem.AddResidualBlock(cost_func, NULL, ksi.data());
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 4;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.FullReport() << std::endl;
    }

    G<T, O> ksi_ = ksi.template cast<T>();
    std::cout << "min pose:" << std::endl << ksi_.matrix() << std::endl;

    return ksi_;
}

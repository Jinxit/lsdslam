#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "two.hpp"
#include "square.hpp"
#include "se3.hpp"
#include "eigen_utils.hpp"
#include "misc_utils.hpp"

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

// Odometry from RGB-D Cameras for Autonomous Quadrocopters (eq 4.22 onwards)
inline Matrix<1, 6> image_pose_jacobian(const studd::two<Image>& gradient,
                                        float x, float y, float inv_depth,
                                        const Eigen::Matrix3f& intrinsic)
{
    Matrix<1, 2> J_I;
    J_I(0, 0) = gradient[0](y, x);
    J_I(0, 1) = gradient[1](y, x);

//    Matrix<1, 2> J_I_;
//    J_I_(0, 0) = gradient[0](y, x) * intrinsic(0, 0);
//    J_I_(0, 1) = gradient[1](y, x) * intrinsic(1, 1);

    Matrix<2, 3> J_pi = Matrix<2, 3>::Zero();
    J_pi(0, 0) =  intrinsic(0, 0) * inv_depth;
    J_pi(1, 1) =  intrinsic(1, 1) * inv_depth;
    J_pi(0, 2) = -intrinsic(0, 0) * x * studd::square(inv_depth);
    J_pi(1, 2) = -intrinsic(1, 1) * y * studd::square(inv_depth);

    Matrix<3, 12> J_g = Matrix<3, 12>::Zero();
    for (int i = 0; i < 3; i++)
    {
        J_g(i, i    ) = x;
        J_g(i, i + 3) = y;
        J_g(i, i + 6) = 1.0 / inv_depth;
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

    /*Matrix<2, 6> J_;
    J_ << 1, 0, -x * inv_depth, -x * y * inv_depth, 1.0 / inv_depth + x * x * inv_depth, -y,
          0, 1, -y * inv_depth, -(1.0 / inv_depth + y * y * inv_depth), x * y * inv_depth, x;
    if (!(J_I_ * J_ * inv_depth - J_I * J_pi * J_g * J_G).isZero(1))
    {
        std::cout << "beep boop" << std::endl;
        std::cout << (J_I_ * J_ * inv_depth) << std::endl << std::endl;
        std::cout << (J_I * J_pi * J_g * J_G) << std::endl;
        exit(1);
    }*/

    return J_I * J_pi * J_g * J_G;
}

inline Eigen::VectorXf calculate_residuals(const Image& warped_image, const Image& ref_image)
{
    auto height = ref_image.rows();
    auto width = ref_image.cols();

    std::vector<float> output;
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            auto warped = warped_image(y, x);

            if (warped >= 0)
            {
                output.push_back(studd::square(ref_image(y, x) - warped));
            }
        }
    }

    return Eigen::Map<Eigen::VectorXf>(output.data(), output.size());
}

inline Eigen::DiagonalMatrix<float, Eigen::Dynamic>
calculate_weights(const Eigen::VectorXf& residuals, const std::function<float(float)>& weighting)
{
    return Eigen::DiagonalMatrix<float, Eigen::Dynamic>(residuals.unaryExpr(weighting));
}

struct warp_error
{
    warp_error(const Eigen::Matrix3f& intrinsic, const Image* const new_image,
               const Image* const ref_image, float x, float y, float inv_depth)
        : new_image(new_image), ref_image(ref_image),
          f_x(intrinsic(0, 0)), f_y(intrinsic(1, 1)),
          c_x(intrinsic(0, 2)), c_y(intrinsic(1, 2)),
          x(x), y(y), inv_depth(inv_depth) { }

    template<class T>
    bool operator()(const T* const ksi, T* residuals) const
    {
        Eigen::Matrix<T, 3, 1> p2(
            T((x - c_x + 0.5f) / (f_x * inv_depth)),
            T((y - c_y + 0.5f) / (f_y * inv_depth)),
            T(1.0 / inv_depth)
        );

        p2 = se3<T>(ksi).exp() * p2;

        T x2 = p2.x() * (T(f_x) / p2.z()) + T(c_x);
        T y2 = p2.y() * (T(f_y) / p2.z()) + T(c_y);

        using Grid = ceres::Grid2D<float, 1, false, false>;
        Grid new_grid(new_image->data(), 0, new_image->rows(), 0, new_image->cols());
        ceres::BiCubicInterpolator<Grid> new_interp(new_grid);

        T new_intensity;
        new_interp.Evaluate(y2, x2, &new_intensity);
        T ref_intensity = T((*ref_image)(int(y), int(x)));
        residuals[0] = T(ref_intensity - new_intensity);
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Matrix3f& intrinsic,
                                       const Image* const new_image,
                                       const Image* const ref_image,
                                       float x, float y, float inv_depth)
    {
        return (new ceres::AutoDiffCostFunction<warp_error, 1, 6>(
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

inline se3<float> ceres_tracking(const sparse_gaussian& sparse_inverse_depth,
                                 const Image& new_image, const Image& ref_image,
                                 const Eigen::Matrix3f& intrinsic,
                                 const std::function<float(float)>& weighting,
                                 const Eigen::Affine3f& guess,
                                 int max_pyramid)
{
    auto height = new_image.rows();
    auto width = new_image.cols();

    se3<float> min_ksi = guess;
    double ksi[6] = {min_ksi.omega[0], min_ksi.omega[1], min_ksi.omega[2],
                     min_ksi.nu[0], min_ksi.nu[1], min_ksi.nu[2]};

    int pyr_i = -1;
    for (int pyr = max_pyramid; pyr > 0; pyr /= 2)
    {
        pyr_i++;

        Eigen::Matrix3f pyramid_intrinsic = intrinsic / pyr;
        pyramid_intrinsic(2, 2) = 1;

        auto ref_resized = pyramid(ref_image, pyr);
        auto new_resized = pyramid(new_image, pyr);

        ceres::Problem problem;
        for (size_t i = 0; i < sparse_inverse_depth.size(); i += pyr)
        {
            auto cost_func = warp_error::create(pyramid_intrinsic,
                                                &new_resized,
                                                &ref_resized,
                                                sparse_inverse_depth[i].first.x() / pyr,
                                                sparse_inverse_depth[i].first.y() / pyr,
                                                sparse_inverse_depth[i].second.mean);
            problem.AddResidualBlock(cost_func, NULL, ksi);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 4;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.FullReport() << std::endl;
        min_ksi = se3<float>(ksi[0], ksi[1], ksi[2], ksi[3], ksi[4], ksi[5]);
        show_residuals("progress", intrinsic, new_image, ref_image, sparse_inverse_depth,
                       min_ksi.exp(), new_image.rows() / pyr,
                       new_image.cols() / pyr, 2 * pyr);
        //cv::waitKey(0);
    }

    std::cout << "min pose:" << std::endl << min_ksi.exp().matrix() << std::endl;

    return min_ksi;
}

inline se3<float> photometric_tracking(const sparse_gaussian& sparse_inverse_depth,
                                       const Image& new_image, const Image& ref_image,
                                       const Eigen::Matrix3f& intrinsic,
                                       const std::function<float(float)>& weighting,
                                       const Eigen::Affine3f& guess,
                                       int max_pyramid)
{
    constexpr size_t max_iterations = 100;
    constexpr float epsilon = 1e-9;

    auto height = new_image.rows();
    auto width = new_image.cols();

    se3<float> ksi = guess;
    se3<float> delta_ksi;
    float last_error = std::numeric_limits<float>::infinity();

    se3<float> min_ksi = guess;
    float min_error = std::numeric_limits<float>::infinity();

    int pyr_i = -1;
    for (int pyr = max_pyramid; pyr > 0; pyr /= 2)
    {
        pyr_i++;
        std::cout << "pyramid:   " << pyr << std::endl;
        if (sparse_inverse_depth.size() == 0)
            continue;

        Eigen::Matrix3f pyramid_intrinsic = intrinsic / pyr;
        pyramid_intrinsic(2, 2) = 1;

        float guess_error = 0;
        for (size_t i = 0; i < max_iterations; i++)
        {
            se3<float> new_ksi = ksi;
            if (i == 0)
                new_ksi = min_ksi;
            sparse_gaussian warped;
            if (i != 0)
            {
                if (delta_ksi.omega[0] != delta_ksi.omega[0])
                {
                    std::cerr << "erroring" << std::endl;
                    std::cerr << sparse_inverse_depth.size() << std::endl;
                    exit(1);
                }
                new_ksi = delta_ksi ^ new_ksi;
                warped = warp(sparse_inverse_depth, pyramid_intrinsic, (-new_ksi).exp());
            }
            else
            {
                warped = sparse_inverse_depth;
            }

            // TODO: MAYBE replace with densify_depth somehow? some overload?
            // TODO: this is dumb as fuck, how can we do gradients better?
            Image warped_image = Image::Constant(height / pyr, width / pyr, -1);
            Image warped_image_inv_depth = Image::Zero(height / pyr, width / pyr);
            Image warped_image_variance = -Image::Ones(height / pyr, width / pyr);
            for (size_t j = 0; j < warped.size(); j++)
            {
                float x = float(warped[j].first.x()) / pyr;
                float y = float(warped[j].first.y()) / pyr;
                Eigen::Vector2f p(x, y);

                if (x > 0 && x < width / pyr - 1 && y > 0 && y < height / pyr - 1)
                {
                    int sx = sparse_inverse_depth[j].first.x() / pyr;
                    int sy = sparse_inverse_depth[j].first.y() / pyr;

                    if (sx > 0 && sx < width / pyr - 1 && sy > 0 && sy < height / pyr - 1)
                    {
                        gaussian warped_pixel = warped[j].second
                                              ^ gaussian(interpolate(warped_image_inv_depth, p),
                                                         interpolate(warped_image_variance, p));
                        if (warped[j].second.mean > warped_image_inv_depth(sy, sx))
                        {
                            if (x * pyr < width && y * pyr < height)
                                warped_image(sy, sx) = interpolate(new_image, p * pyr);
                        }
                        warped_image_inv_depth(sy, sx) = warped_pixel.mean;
                        warped_image_variance(sy, sx) = warped_pixel.variance;
                    }
                }
            }
            show("warped", warped_image);
            //cv::waitKey(500);

            Image ref_resized = Image::Zero(height / pyr, width / pyr);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ref_resized(y / pyr, x / pyr) += ref_image(y, x) / studd::square(pyr);
                }
            }

            Eigen::VectorXf residuals = calculate_residuals(warped_image, ref_resized);
            Eigen::DiagonalMatrix<float, Eigen::Dynamic> weights = calculate_weights(residuals, weighting);
            float error = ((residuals.transpose() * weights).dot(residuals))
                        / residuals.size();

            // consider switching to (last_error - error) / last_error < eps, according to wiki
            //if (error > last_error && i != 0)
            
            if (i == 0)
            {
                guess_error = error;
                min_error = error;
            }
            else if (error - last_error > 0)
            {
                std::cout << "berror: " << error << std::endl;
                std::cout << "BOOM" << std::endl;
                break;
            }
            else if (error < min_error)
            {
                min_ksi = new_ksi;
                min_error = error;
            }
            /*if (error < epsilon)
            {
                std::cout << "error is zero yo" << std::endl;
                break;
            }*/

            studd::two<Image> gradient = sobel(warped_image);

            std::vector<Eigen::Vector2i> J_helper;
            Eigen::MatrixXf J = Eigen::MatrixXf(residuals.size(), 6);
            {
                size_t j = 0;
                for (size_t y = 0; y < height / pyr; y++)
                {
                    for (size_t x = 0; x < width / pyr; x++)
                    {
                        if (warped_image(y, x) >= 0)
                        {
                            J.row(j) = image_pose_jacobian(gradient, x, y,
                                                           warped_image_inv_depth(y, x),
                                                           pyramid_intrinsic);
                            J_helper.emplace_back(x, y);
                            j++;
                        }
                    }
                }
            }

            //show_jacobian("J", J, J_helper, height / pyr, width / pyr, pyr * 2);

            Eigen::MatrixXf J_t_weights = J.transpose() * weights;
            ksi = new_ksi;

            delta_ksi = -se3<float>((J_t_weights * J).inverse() * J_t_weights * residuals);
            //delta_ksi = se3((J_t_weights * J).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
            //                .solve(-J_t_weights * residuals).topLeftCorner<6, 1>());

            //std::cout << "ksi:       " << ksi << std::endl;
            //std::cout << "delta_ksi: " << delta_ksi << std::endl;
            std::cout << "error: " << error << std::endl;
            //std::cout << "min_ksi: " << min_ksi << std::endl;
            std::cout << "min_error: " << min_error << std::endl;
            std::cout << "guess_error: " << guess_error << std::endl;
            //if (std::abs(error - last_error) < epsilon)
            //    break;

            last_error = error;

            show_residuals("in-progress", new_image, ref_image, sparse_inverse_depth, warped,
                           height / pyr, width / pyr, pyr * 4);
            cv::waitKey(i == 0 ? 0 : 10);// * std::pow<double>(pyr, 1.5));
        }
    }

    std::cout << "min pose:" << std::endl << min_ksi.exp().matrix() << std::endl;

    return min_ksi;
}

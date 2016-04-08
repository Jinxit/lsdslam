#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "two.hpp"
#include "square.hpp"
#include "se3.hpp"
#include "eigen_utils.hpp"
#include "misc_utils.hpp"

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
inline Matrix<1, 6> image_pose_jacobian(const studd::two<Image>& gradient,
                                        float x, float y, float inverse_depth,
                                        const Eigen::Matrix3f& intrinsic, int pyr)
{
    Matrix<1, 2> J_I;
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
        if (inverse_depth == 0)
        {
            std::cout << x << " " << y << " " << pyr << std::endl << intrinsic << std::endl;
            exit(1);
        }
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

inline std::vector<pixel> warp(std::vector<pixel> sparse_inverse_depth,
                               const Eigen::Matrix3f& intrinsic,
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

inline std::pair<Eigen::VectorXf, Eigen::DiagonalMatrix<float, Eigen::Dynamic>>
residuals_and_weights(const std::vector<pixel>& sparse_inverse_depth,
                      const std::vector<pixel>& warped,
                      const Image& new_image, const Image& ref_image,
                      const std::function<float(float)>& weighting)
{
    Eigen::VectorXf residuals(sparse_inverse_depth.size());
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
            residuals(i) = 0;
            weights.diagonal()[i] = 0;
        }
    }

    return {residuals, weights};
}

inline se3 photometric_tracking(const std::vector<std::vector<pixel>>& sparse_inverse_depth,
                                const Image& new_image, const Image& ref_image,
                                const Eigen::Matrix3f& intrinsic,
                                const std::function<float(float)>& weighting)
{
    constexpr size_t max_iterations = 10;
    constexpr float epsilon = 1e-9;

    auto height = new_image.rows();
    auto width = new_image.cols();
    auto max_pyramid = std::pow(2, sparse_inverse_depth.size() - 1);

    se3 ksi;
    se3 delta_ksi;
    float last_error = std::numeric_limits<float>::infinity();

    int pyr_i = -1;
    for (int pyr = max_pyramid; pyr > 0; pyr /= 2)
    {
        pyr_i++;
        std::cout << "pyr:   " << pyr << std::endl;
        std::cout << "pyr_i: " << pyr_i << std::endl;
        if (sparse_inverse_depth[pyr_i].size() == 0)
            continue;

        for (size_t i = 0; i < max_iterations; i++)
        {
            se3 new_ksi;
            std::vector<pixel> warped;
            if (i != 0 || pyr != max_pyramid)
            {
                if (delta_ksi.omega[0] != delta_ksi.omega[0])
                {
                    std::cerr << "erroring" << std::endl;
                    std::cerr << sparse_inverse_depth[pyr_i].size() << std::endl;
                    exit(1);
                }
                new_ksi = ksi ^ delta_ksi;
                warped = warp(sparse_inverse_depth[pyr_i], intrinsic, new_ksi.exp());
            }
            else
            {
                warped = sparse_inverse_depth[pyr_i];
            }

            // TODO: this is dumb as fuck, how can we do gradients better?
            Image warped_image = Image::Zero(height / pyr, width / pyr);
            Image warped_image_inv_depth = Image::Zero(height / pyr, width / pyr);
            for (size_t i = 0; i < warped.size(); i++)
            {
                int x = warped[i].x / pyr;
                int y = warped[i].y / pyr;
                if (x >= 0 && x < width / pyr && y >= 0 && y < height / pyr)
                {
                    int sx = sparse_inverse_depth[pyr_i][i].x;
                    int sy = sparse_inverse_depth[pyr_i][i].y;

                    if (warped_image_inv_depth(y, x) < warped[i].inverse_depth)
                    {
                        warped_image_inv_depth(y, x) = warped[i].inverse_depth;
                        warped_image(y, x) = ref_image(sy, sx);
                    }
                }
            }

            studd::two<Image> gradient = sobel(warped_image);

            Eigen::VectorXf residuals;
            Eigen::DiagonalMatrix<float, Eigen::Dynamic> weights;
            std::tie(residuals, weights) = residuals_and_weights(sparse_inverse_depth[pyr_i], warped,
                                                                 new_image, ref_image, weighting);

            float error = ((weights * residuals).dot(residuals)
                        / sparse_inverse_depth[pyr_i].size());

            // consider switching to (last_error - error) / last_error < eps, according to wiki
            if (error > last_error)
            //if (error - last_error > 1e-2)
            {
                std::cout << "berror: " << error << std::endl;
                std::cout << "BOOM" << std::endl;
                delta_ksi = se3();
                break;
            }
            if (error < epsilon)
            {
                std::cout << "error is zero yo" << std::endl;
                break;
            }

            Eigen::MatrixXf J = Eigen::MatrixXf(warped.size(), 6);
            for (size_t i = 0; i < sparse_inverse_depth[pyr_i].size(); i++)
            {
                auto& p = sparse_inverse_depth[pyr_i][i];
                J.row(i) = image_pose_jacobian(gradient, p.x, p.y, p.inverse_depth, intrinsic, pyr);
            }

            Eigen::MatrixXf J_t_weights = J.transpose() * weights;
            ksi = new_ksi;
            //delta_ksi = se3((J_t_weights * J).inverse() * J_t_weights * residuals);
            delta_ksi = se3((J_t_weights * J).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                            .solve(J_t_weights * residuals).topLeftCorner<6, 1>());

            std::cout << "ksi:       " << ksi << std::endl;
            std::cout << "delta_ksi: " << delta_ksi << std::endl;
            std::cout << "error: " << error << std::endl;
            if (std::abs(error - last_error) < epsilon)
                break;

            last_error = error;
        }
    }

    return ksi;
}

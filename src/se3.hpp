#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include "eigen_utils.hpp"

struct se3
{
    se3() : omega(Eigen::Vector3f::Zero()), nu(Eigen::Vector3f::Zero()) { }
    se3(const Eigen::Vector3f& omega, const Eigen::Vector3f& nu) : omega(omega), nu(nu) { }
    se3(const Eigen::Matrix<float, 6, 1>& m)
        : omega(m.topLeftCorner<3, 1>()), nu(m.topRightCorner<3, 1>()) { }
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
                          + ((w_norm - std::sin(w_norm)) / (w_norm * w_norm2)) * skew2;

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
        stream << self.omega[0] << " "
               << self.omega[1] << " "
               << self.omega[2] << " "
               << self.nu[0] << " "
               << self.nu[1] << " "
               << self.nu[2];
        return stream;
    }

    Eigen::Vector3f omega;
    Eigen::Vector3f nu;
};
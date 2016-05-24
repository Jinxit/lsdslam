#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include "eigen_utils.hpp"

template<class T>
struct se3
{
    using Affine = Eigen::Transform<T, 3, Eigen::Affine>;

    se3() : omega(Eigen::Matrix<T, 3, 1>::Zero()), nu(Eigen::Matrix<T, 3, 1>::Zero()) { }
    se3(const Eigen::Matrix<T, 3, 1>& omega, const Eigen::Matrix<T, 3, 1>& nu)
        : omega(omega), nu(nu) { }
    se3(const Eigen::Matrix<T, 6, 1>& m)
        : omega(m.template topLeftCorner<3, 1>()), nu(m.template topRightCorner<3, 1>()) { }
    se3(T omega_x, T omega_y, T omega_z, T nu_x, T nu_y, T nu_z)
        : omega({omega_x, omega_y, omega_z}), nu({nu_x, nu_y, nu_z}) { }
    se3(const Affine& noncanonical)
    {
        Eigen::Matrix<T, 4, 4> logged = noncanonical.matrix().log();
        omega = unskewed<T>(logged.template topLeftCorner<3, 3>());
        nu = logged.template topRightCorner<3, 1>();
    }

    Affine exp() const
    {
        auto skew = skewed<T>(omega);
        auto skew2 = studd::square(skew);
        auto w_norm2 = std::max(T(1e-8), omega.squaredNorm()); // because division by zero, yo
        auto w_norm = sqrt(w_norm2);

        Eigen::Matrix<T, 3, 3> e_omega = Eigen::Matrix<T, 3, 3>::Identity()
                                       + (sin(w_norm) / w_norm) * skew
                                       + ((T(1) - cos(w_norm)) / w_norm2) * skew2;
        Eigen::Matrix<T, 3, 3> V = Eigen::Matrix<T, 3, 3>::Identity()
                                 + ((T(1) - cos(w_norm)) / w_norm2) * skew
                                 + ((w_norm - sin(w_norm)) / (w_norm * w_norm2)) * skew2;

        Eigen::Matrix<T, 4, 4> output = Eigen::Matrix<T, 4, 4>();
        output.template topLeftCorner<3, 3>() = e_omega;
        output.template topRightCorner<3, 1>() = V * nu;
        output(3, 3) = T(1);

        return Affine(output);
    }

    friend se3 operator^(const se3& ksi, const se3& ksi_delta)
    {
        return se3(ksi.exp() * ksi_delta.exp());
    }

    friend se3 operator-(const se3& ksi)
    {
        se3 output;
        output.omega = -ksi.omega;
        output.nu = -ksi.nu;
        return output;
    }

    /*
    friend std::ostream& operator<< (std::ostream& stream, const se3& self) {
        stream << self.omega[0] << " "
               << self.omega[1] << " "
               << self.omega[2] << " "
               << self.nu[0] << " "
               << self.nu[1] << " "
               << self.nu[2];
        return stream;
    }*/

    Eigen::Matrix<T, 3, 1> omega;
    Eigen::Matrix<T, 3, 1> nu;
};
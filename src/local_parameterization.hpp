#pragma once

// stolen from stevenlovegrove/Sophus tests
// but generalized by Jinxit

#include <ceres/local_parameterization.h>

namespace Sophus
{
    template<class G, bool LeftMultiply = false>
    class LocalParameterization : public ceres::LocalParameterization
    {
        static constexpr int DoF = G::DoF;
        static constexpr int num_parameters = G::num_parameters;
    public:
        virtual ~LocalParameterization() {}

        /**
         * \brief SE3 plus operation for Ceres
         *
         * \f$ T\cdot\exp(\widehat{\delta}) \f$
         */
        virtual bool Plus(const double * T_raw, const double * delta_raw,
                          double * T_plus_delta_raw) const
        {
            const Eigen::Map<const G> T(T_raw);
            const Eigen::Map<const Eigen::Matrix<double, DoF, 1>> delta(delta_raw);
            Eigen::Map<G> T_plus_delta(T_plus_delta_raw);

            if (LeftMultiply)
            {
                T_plus_delta = G::exp(delta) * T;
            }
            else
            {
                T_plus_delta = T * G::exp(delta);
            }

            return true;
        }

        /**
         * \brief Jacobian of SE3 plus operation for Ceres
         *
         * \f$ \frac{\partial}{\partial \delta}T\cdot\exp(\widehat{\delta})|_{\delta=0} \f$
         */
        virtual bool ComputeJacobian(const double * T_raw, double * jacobian_raw) const
        {
            const Eigen::Map<const G> T(T_raw);
            Eigen::Map<Eigen::Matrix<double, DoF, num_parameters>> jacobian(jacobian_raw);
            jacobian = T.internalJacobian().transpose();
            return true;
        }

        virtual int GlobalSize() const
        {
            return num_parameters;
        }

        virtual int LocalSize() const
        {
            return DoF;
        }
    };
}

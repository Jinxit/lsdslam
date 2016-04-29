#pragma once

#include "base_tracker.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "../two.hpp"
#include "../eigen_utils.hpp"
#include "../disparity.hpp"
#include "../epiline.hpp"

namespace stereo
{
    class tracker : public base_tracker<studd::two<Image>>
    {
    public:
        tracker(const Eigen::Affine3f& pose,
                const Eigen::Vector2i& resolution,
                const std::function<float(float)>& weighting,
                const Eigen::Matrix3f& static_fundamental,
                const Eigen::Affine3f& static_transform,
                const Eigen::Matrix3f& left_intrinsic,
                const Eigen::Affine3f& left_transform);

        Eigen::Affine3f update(const studd::two<Image>& observation, const Eigen::Affine3f& guess);

    private:
        studd::two<Image> static_stereo(const Image& left, const Image& right,
                                        const studd::two<Image>& gradient);

        Eigen::Matrix3f static_fundamental;
        Eigen::Affine3f static_transform;
        Eigen::Matrix3f left_intrinsic;
        Eigen::Affine3f left_transform;
        std::vector<epiline> stereo_epilines;
        Eigen::Vector2f stereo_epipole;
    };
}
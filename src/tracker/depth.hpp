#pragma once

#include "base_tracker.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "../two.hpp"
#include "../eigen_utils.hpp"
#include "../disparity.hpp"
#include "../epiline.hpp"
#include "../dynamic_map.hpp"

namespace depth
{
    class tracker : public base_tracker<studd::two<Image>>
    {
    public:
        tracker(const Sophus::SE3f& pose,
                const Eigen::Vector2i& resolution,
                const std::function<float(float)>& weighting,
                const Eigen::Matrix3f& intrinsic,
                const studd::two<Image>& observation);

        Sophus::SE3f update(const studd::two<Image>& observation, const Sophus::SE3f& guess);

    private:
        studd::two<Image> filter_depth(studd::two<Image> depth, const Image& intensity);

        Eigen::Matrix3f intrinsic;
    };
}
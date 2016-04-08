#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "two.hpp"
#include "eigen_utils.hpp"
#include "disparity.hpp"
#include "epiline.hpp"
#include "loader.hpp"

struct keyframe
{
    keyframe(int height, int width)
        : left(height, width), right(height, width),
          inverse_depth(Image(height, width), Image(height, width))
    { }

    Image left;
    Image right; // unnecessary?
    studd::two<Image> inverse_depth;
    Eigen::Affine3f pose;
};

class tracker
{
public:
    tracker(const stereo_calibration& sc, const Eigen::Affine3f& pose,
            const Image& new_left, const Image& new_right);

    Eigen::Affine3f update(const Image& new_left, const Image& new_right);
    Eigen::Affine3f get_pose() const { return pose; }

private:
    studd::two<Image> static_stereo(const Image& left, const Image& right,
                                    const studd::two<Image>& gradient);
    studd::two<Image> temporal_stereo(const Image& left, const Image& right,
                                      const studd::two<Image>& gradient,
                                      const Eigen::Affine3f& transform);

    Eigen::Affine3f pose;
    stereo_calibration sc;
    keyframe kf;
    std::vector<epiline> stereo_epilines;
    Eigen::Vector2f stereo_epipole;
    std::function<float(float)> weighting;
};
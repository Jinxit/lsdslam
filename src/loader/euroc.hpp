#pragma once

#include "base_loader.hpp"

#include <utility>
#include <vector>
#include <array>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "opencv2/core/core.hpp"

#include "../se3.hpp"
#include "../dynamic_map.hpp"

namespace euroc
{
    struct frame
    {
        Image left;
        Image right;
        Eigen::Affine3f pose;
    };

    // brown's distortion model
    struct distortion_parameters
    {
        std::array<float, 2> k;
        std::array<float, 2> p;

        std::vector<float> as_vector() const
        {
            return { k[0], k[1], p[0], p[1] };
        }
    };

    struct calibration
    {
        distortion_parameters distortion;
        Eigen::Matrix3f intrinsic;
        Eigen::Affine3f extrinsic;
        cv::Mat map_x;
        cv::Mat map_y;
    };

    struct stereo_calibration
    {
        calibration left;
        calibration right;
        Eigen::Matrix3f static_fundamental;
        Eigen::Vector2i resolution;
        Eigen::Affine3f transform_left_right;
        Eigen::Affine3f transform_left;
    };

    class loader : public base_loader<frame, stereo_calibration>
    {
    public:
        loader(const std::string& folder);

        frame operator[](size_t i) override;

    private:
        Eigen::Affine3f pose_at(size_t i);
        void recalibrate();

        std::string folder;
        std::vector<size_t> indices;
        std::vector<std::pair<size_t, Eigen::Affine3f>> poses;
        studd::dynamic_map<size_t, Image> left_map;
        studd::dynamic_map<size_t, Image> right_map;
    };
    
}

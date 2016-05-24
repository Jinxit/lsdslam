#pragma once

#include "base_loader.hpp"

#include <utility>
#include <vector>
#include <array>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "../se3.hpp"
#include "../dynamic_map.hpp"
#include "../two.hpp"

namespace tum
{
    struct frame
    {
        Image intensity;
        Image depth;
        Eigen::Affine3f pose;
    };

    struct calibration
    {
        Eigen::Matrix3f intrinsic;
        Eigen::Vector2i resolution;
    };

    class loader : public base_loader<frame, calibration>
    {
    public:
        loader(const std::string& folder);

        frame operator[](size_t i) override;

    private:
        Eigen::Affine3f pose_at(const std::string& i);

        std::string folder;
        std::vector<studd::two<std::string>> indices;
        std::vector<std::pair<double, Eigen::Affine3f>> poses;
        studd::dynamic_map<std::string, Image> intensity_map;
        studd::dynamic_map<std::string, Image> depth_map;
    };
    
}

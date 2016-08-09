#pragma once

#include "base_loader.hpp"

#include <utility>
#include <vector>
#include <array>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "../dynamic_map.hpp"
#include "../two.hpp"
#include "../eigen_utils.hpp"

namespace tum
{
    struct frame
    {
        Image intensity;
        Image depth;
        Sophus::SE3f pose;
        std::string timestamp;
    };

    struct calibration
    {
        Eigen::Matrix3f intrinsic = Eigen::Matrix3f::Zero();
        Eigen::Vector2i resolution;
    };

    class loader : public base_loader<frame, calibration>
    {
    public:
        loader(const std::string& folder);

        frame operator[](size_t i) override;
        size_t size() const override;

    private:
        std::pair<std::string, Sophus::SE3f> pose_at(const std::string& i);

        std::string folder;
        std::vector<studd::two<std::string>> indices;
        std::vector<std::pair<double, Sophus::SE3f>> poses;
        studd::dynamic_map<std::string, Image> intensity_map;
        studd::dynamic_map<std::string, Image> depth_map;
    };
    
}

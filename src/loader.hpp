#pragma once

#include <utility>
#include <vector>
#include <iostream>
#include <algorithm>
#include <array>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <yaml-cpp/yaml.h>

#include "se3.hpp"
#include "eigen_utils.hpp"
#include "dynamic_map.hpp"

namespace fs = boost::filesystem;

using Image = Eigen::MatrixXf;

struct frame
{
    Image left;
    Image right;
    Eigen::Affine3f pose;
};

struct stereo_frame
{
    frame before;
    frame after;
    se3 transform;
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
};

class loader
{
public:
    loader(const std::string& folder);

    stereo_frame operator()(size_t i, size_t j);
    frame operator[](size_t i);
    stereo_calibration get_calibration() const { return c; }

private:
    std::string folder;
    stereo_calibration c;
    std::vector<size_t> indices;
    std::vector<std::pair<size_t, Eigen::Affine3f>> poses;
    studd::dynamic_map<size_t, Image> left_map;
    studd::dynamic_map<size_t, Image> right_map;

    Eigen::Affine3f pose_at(size_t i);
};

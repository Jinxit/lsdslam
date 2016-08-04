#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <sophus/se3.hpp>

#include "eigen_utils.hpp"
#include "square.hpp"

struct epiline
{
    epiline(const Eigen::Vector2i point, const Eigen::Vector3f line)
        : point(point), line(line) { };

    Eigen::Vector2i point;
    Eigen::Vector3f line;
};


inline epiline generate_epiline(const Eigen::Vector2i& p,
                                const Eigen::Matrix3f& fundamental, int height, int width)
{
    /*
    std::vector<cv::Point2f> in{cv::Point2f(p.x(), p.y())};
    std::vector<cv::Point3f> out;
    cv::Mat F;
    cv::eigen2cv(fundamental, F);
    cv::computeCorrespondEpilines(in, 2, F, out);
    return epiline(p, Eigen::Vector3f(out[0].x, out[0].y, out[0].z));
    */

    Eigen::Vector3f line = fundamental * Eigen::Vector3f(p.x(), p.y(), 1);
    auto nu = studd::square(line[0]) + studd::square(line[1]);
    if (nu != 0 && !(nu != nu))
    {
        //line /= std::sqrt(nu);
    }

    return epiline(p, line);
}

inline std::vector<epiline> generate_epilines(int height, int width,
                                              const Eigen::Matrix3f& fundamental)
{
    std::vector<epiline> epilines;
    epilines.reserve(height * width);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            epilines.emplace_back(generate_epiline(Eigen::Vector2i(x, y), fundamental, height, width));
        }
    }

    return epilines;
}

inline Eigen::Vector2f generate_epipole(const Sophus::SE3f& transform,
                                        const Eigen::Matrix3f new_intrinsic)
{
    Eigen::Vector3f homogenous = new_intrinsic * transform.translation();
    return homogenous.topLeftCorner<2, 1>() / homogenous[2];
}

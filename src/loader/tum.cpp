#include "tum.hpp"

#include <utility>
#include <vector>
#include <iostream>
#include <algorithm>
#include <array>
#include <iomanip>
#include <fstream>

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

#include "../eigen_utils.hpp"

namespace fs = boost::filesystem;

namespace tum
{
    constexpr float image_rescaling = 2.0;

    namespace
    {
        std::function<Image(const std::string&)>
        create_intensity_loader(const std::string& folder)
        {
            return [folder](const std::string& id) {
                cv::Mat image = cv::imread(folder + id + ".png",
                                           CV_LOAD_IMAGE_ANYDEPTH);

                cv::resize(image, image, cv::Size(), 1.0 / image_rescaling, 1.0 / image_rescaling);

                Image output = Image::Zero(image.rows, image.cols);
                for (int y = 0; y < image.rows; y++)
                {
                    for (int x = 0; x < image.cols; x++)
                    {
                        output(y, x) = image.at<unsigned char>(y, x) / 255.0;
                    }
                }

                return output;
            };
        }

        std::function<Image(const std::string&)>
        create_depth_loader(const std::string& folder)
        {
            return [folder](const std::string& id) {
                cv::Mat image = cv::imread(folder + id + ".png",
                                           CV_LOAD_IMAGE_ANYDEPTH);

                cv::resize(image, image, cv::Size(), 1.0 / image_rescaling, 1.0 / image_rescaling);

                Image output = Image::Zero(image.rows, image.cols);
                for (int y = 0; y < image.rows; y++)
                {
                    for (int x = 0; x < image.cols; x++)
                    {
                        output(y, x) = image.at<unsigned short>(y, x) / 5000.0;
                    }
                }

                return output;
            };
        }

        bool timestamp_search(const std::pair<double, Sophus::SE3f>& pose, double i)
        {
            return pose.first < i;
        }

        std::vector<studd::two<std::string>> load_indices(const std::string& folder)
        {
            std::vector<studd::two<std::string>> output;
            std::ifstream synced_stream(folder + "synced.txt");

            std::string line;
            std::string skip;
            while(std::getline(synced_stream, line))
            {
                std::stringstream line_stream(line);
                std::string rgb_time;
                std::string depth_time;
                line_stream >> rgb_time >> skip >> depth_time;

                output.push_back({rgb_time, depth_time}); 
            }
            return output;
        }

        std::vector<std::pair<double, Sophus::SE3f>> load_poses(const std::string& folder)
        {
            std::ifstream pose_stream(folder + "groundtruth.txt");

            std::string line;
            std::vector<std::pair<double, Sophus::SE3f>> poses;

            // eat the first three lines, hacky
            std::getline(pose_stream, line);
            std::getline(pose_stream, line);
            std::getline(pose_stream, line);
            while (std::getline(pose_stream, line))
            {
                std::stringstream line_stream(line);

                double timestamp;
                line_stream >> timestamp;

                Eigen::Vector3f pos;
                line_stream >> pos.x() >> pos.y() >> pos.z();

                Eigen::Quaternionf rot;
                line_stream >> rot.x() >> rot.y() >> rot.z() >> rot.w();

                Sophus::SE3f pose;
                pose.translation() = pos;
                pose.setQuaternion(rot);
                poses.emplace_back(timestamp, pose);
            }
            return poses;
        }

        calibration load_calibration(const std::string& folder)
        {
            calibration output;

            output.intrinsic(0, 0) = 525 / image_rescaling;
            output.intrinsic(1, 1) = 525 / image_rescaling;
            output.intrinsic(0, 2) = 319.5 / image_rescaling;
            output.intrinsic(1, 2) = 239.5 / image_rescaling;
            output.intrinsic(2, 2) = 1;

            output.resolution.x() = 640 / image_rescaling;
            output.resolution.y() = 480 / image_rescaling;

            return output;
        }
    }

    loader::loader(const std::string& folder)
        : base_loader(load_calibration(folder)), folder(folder),
          indices(load_indices(folder)),
          poses(load_poses(folder)),
          intensity_map(create_intensity_loader(folder + "rgb/")),
          depth_map(create_depth_loader(folder + "depth/"))
    {
    }

    frame loader::operator[](size_t i)
    {
        frame output;
        output.intensity = intensity_map[indices[i].first];
        output.depth = depth_map[indices[i].second];
        std::tie(output.timestamp, output.pose) = pose_at(indices[i].first);

        return output;
    }


    std::pair<std::string, Sophus::SE3f> loader::pose_at(const std::string& i_str)
    {
        double i = std::stod(i_str);
        auto it_after = std::lower_bound(poses.begin(), poses.end(),
                                         i, &timestamp_search);
        auto it_before = it_after;
        if (it_after->first == i || it_before == poses.begin())
        {
            ++it_after;
        }
        else
        {
            --it_before;
        }

        double t = (i - it_before->first) / (it_after->first - it_before->first);
        auto interp = interpolate(it_before->second, it_after->second, t);
        return {i_str, interp};
    }

    size_t loader::size() const
    {
        return indices.size();
    }
}
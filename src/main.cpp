#include <iostream>
#include <string>
#include <utility>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/eigen.hpp"

#include <sophus/se3.hpp>

#include "glog/logging.h"

#include "two.hpp"
#include "square.hpp"
#include "eigen_utils.hpp"
#include "loader/euroc.hpp"
#include "loader/tum.hpp"
#include "disparity.hpp"
#include "epiline.hpp"
#include "gaussian.hpp"
#include "tracker/stereo.hpp"
#include "tracker/depth.hpp"

void report(const Eigen::Affine3f& pose)
{
    std::cout << (Eigen::AngleAxisf(pose.rotation()).angle() * 180.0f / M_PI)
              << " degrees and " << pose.translation().norm() << "m" << std::endl;
}

int main(int argc, const char* argv[])
{
    google::InitGoogleLogging(argv[0]);

    auto data = tum::loader("data/TUM/freiburg2_xyz/");
    auto sc = data.get_calibration();

    unsigned int frame_skip = 1;
    unsigned int start_offset = 0;

    auto first_frame = data[start_offset];
    //auto t = stereo::tracker(first_frame.pose, sc.resolution, [](float x) { return 1; },
    //                         sc.static_fundamental, sc.transform_left_right,
    //                         sc.left.intrinsic, sc.transform_left);
    auto t = depth::tracker(first_frame.pose, sc.resolution, sc.intrinsic,
                            {first_frame.intensity, first_frame.depth});

    for (size_t i = start_offset + frame_skip; i < 10000; i += frame_skip)
    {
        //std::cout << "original pose:" << std::endl << t.get_pose().matrix() << std::endl;
        auto new_frame = data[i];
        auto guess = data[i - frame_skip].pose.inverse() * new_frame.pose;
        auto start = std::chrono::steady_clock::now();
        auto travelled = t.update({new_frame.intensity, new_frame.depth}, new_frame.pose);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration<double, std::milli>(end - start).count();
        //std::cerr << ms << " ms" << std::endl;
        std::cerr << (1000.0 / ms) << " fps" << std::endl;
        //std::cout << "actual pose:" << std::endl << new_frame.pose.matrix() << std::endl;
        //std::cout << "tracked pose:" << std::endl << t.get_pose().matrix() << std::endl;
        //std::cout << "diff: " << std::endl;
        //report(t.get_pose().inverse() * new_frame.pose);
        //std::cout << "correct travelled: " << std::endl;
        //report(guess);
        //std::cout << "we travelled: " << std::endl;
        //report(travelled);
        cv::waitKey(1);
    }
}

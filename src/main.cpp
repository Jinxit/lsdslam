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

#include "two.hpp"
#include "square.hpp"
#include "se3.hpp"
#include "eigen_utils.hpp"
#include "loader.hpp"
#include "disparity.hpp"
#include "epiline.hpp"
#include "gaussian.hpp"
#include "tracker.hpp"

void report(const Eigen::Affine3f& pose)
{
    std::cout << (Eigen::AngleAxisf(pose.rotation()).angle() * 180.0f / M_PI)
              << " degrees and " << pose.translation().norm() << "m" << std::endl;
}

int main(int argc, const char* argv[])
{
    auto data = loader("data/MH_01_easy/");
    auto sc = data.get_calibration();

    unsigned int frame_skip = 1;
    unsigned int start_offset = 1000;

    auto height = sc.resolution.y();
    auto width = sc.resolution.x();

    auto first_frame = data[start_offset];
    auto t = tracker(sc, first_frame.pose, first_frame.left, first_frame.right);
    for (size_t i = start_offset + 1; i < 10000; i += frame_skip)
    {
        auto start = std::chrono::steady_clock::now();
        auto new_frame = data[i];
        auto travelled = t.update(new_frame.left, new_frame.right);
        auto end = std::chrono::steady_clock::now();
        auto dt = end - start;
        std::cerr << (std::chrono::duration<double, std::milli>(dt).count()) << " ms" << std::endl;
        std::cout << "actual pose:" << std::endl << new_frame.pose.matrix() << std::endl;
        std::cout << "tracked pose:" << std::endl << t.get_pose().matrix() << std::endl;
        std::cout << "diff: " << std::endl;
        report(t.get_pose().inverse() * new_frame.pose);
        std::cout << "correct travelled: " << std::endl;
        report(data[i - frame_skip].pose.inverse() * new_frame.pose);
        std::cout << "we travelled: " << std::endl;
        report(travelled);
    }
}

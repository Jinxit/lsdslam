#include "depth.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <sophus/se3.hpp>

#include "../gaussian.hpp"
#include "../two.hpp"
#include "../photometric_tracking.hpp"
#include "../square.hpp"
#include "../misc_utils.hpp"

namespace depth
{
    tracker::tracker(const Sophus::SE3f& pose,
                     const Eigen::Vector2i& resolution,
                     const std::function<float(float)>& weighting,
                     const Eigen::Matrix3f& intrinsic,
                     const studd::two<Image>& observation)
        : base_tracker(pose, resolution, weighting),
          intrinsic(intrinsic)
    {
        kf.intensity = observation[0];
        kf.inverse_depth = filter_depth(studd::two<Image>(observation[1],
                                                          Image::Ones(resolution.y(), resolution.x())));
    }

    studd::two<Image> tracker::filter_depth(studd::two<Image> depth)
    {
        for (int y = 0; y < resolution.y(); y++)
        {
            for (int x = 0; x < resolution.x(); x++)
            {
                if (depth[0](y, x) == 0)
                {
                    depth[1](y, x) = -1;
                }
            }
        }

        return depth;
    }

    Sophus::SE3f tracker::update(const studd::two<Image>& observation, const Sophus::SE3f& guess)
    {
        constexpr float keyframe_distance = 0.2;
        constexpr float keyframe_angle = M_PI / 8;

        auto& new_intensity = observation[0];
        auto new_depth = studd::two<Image>(observation[1], Image::Ones(resolution.y(), resolution.x()));

        //show_rainbow("before", kf.inverse_depth, observation[0]);
        new_depth = filter_depth(new_depth);
        //show_rainbow("new_depth", new_depth, observation[0]);
        auto sparse_depth = sparsify_depth(new_depth, true);

        //play(sparse_depth, new_intensity, kf.intensity, intrinsic);

        auto current_to_kf_before = kf.pose.inverse() * pose;
        auto uh = kf.pose.inverse() * pose * guess;
        Sophus::SE3f transform = photometric_tracking<Sophus::SE3Group>(
            sparse_depth,
            new_intensity, kf.intensity,
            intrinsic,
            weighting,
            current_to_kf_before,
            32);
        show_residuals("minimized", intrinsic, new_intensity, kf.intensity, sparse_depth,
                       transform, resolution.y(), resolution.x(), 2);
        std::cout << "debug: " << std::endl;
        std::cout << current_to_kf_before.matrix() << std::endl << std::endl;
        std::cout << transform.matrix() << std::endl;
        transform = guess;
        pose = pose * transform;
        Sophus::SE3f current_to_kf = kf.pose.inverse() * pose;
        show_residuals("theoretical", intrinsic, new_intensity, kf.intensity, sparse_depth,
                       current_to_kf.inverse(), resolution.y(), resolution.x(), 2);
        show_residuals("starting point", intrinsic, new_intensity, kf.intensity, sparse_depth,
                       current_to_kf_before.inverse(), resolution.y(), resolution.x(), 2);
        //show_residuals("nothing", intrinsic, new_intensity, kf.intensity, sparse_depth,
        //               Eigen::Affine3f(Eigen::Matrix4f::Identity()), resolution.y(), resolution.x(), 2);
        auto warped_depth = regularize_depth(densify_depth(warp(sparse_depth, intrinsic,
                                                                current_to_kf),
                                             resolution.y(), resolution.x()));
        //show_rainbow("warped_depth", warped_depth, observation[0]);
        kf.inverse_depth = fuse_depth(kf.inverse_depth, warped_depth);
        //show_rainbow("fused", kf.inverse_depth, observation[0]);
        cv::waitKey(0);

        /*
        if (false && (kf_to_current.translation().norm() > keyframe_distance ||
            Eigen::AngleAxisf(kf_to_current.rotation()).angle() > keyframe_angle))
        {
            std::cout << transform.translation() << std::endl;
            std::cout << guess.translation() << std::endl;
            std::cout << "new keyframe" << std::endl;
            // initialize keyframe
            kf.inverse_depth = regularize_depth(densify_depth(warp(sparsify_depth(kf.inverse_depth, 1)[0],
                                                                   left_intrinsic, kf_to_current),
                                                              resolution.y(), resolution.x()));
            kf.pose = pose;
            kf.intensity = observation[0];
        }
        else
        {
            // TODO: this might need some warping OR just reversing new/ref
            //auto t_stereo = temporal_stereo(observation[0], kf.left, gradient, transform);
            //show_rainbow("t_stereo", t_stereo, observation[0]);
            //kf.inverse_depth = fuse_depth(kf.inverse_depth, t_stereo);
            //show_rainbow("fused_temporal_static", kf.inverse_depth, observation[0]);
            //cv::waitKey(0);
            //cv::destroyWindow("fused_temporal_static");
            kf.inverse_depth = regularize_depth(kf.inverse_depth);
            show_rainbow("regularized", kf.inverse_depth, observation[0]);
            cv::waitKey(0);
        }*/
        //play(observation[0], observation[1]);

        return transform;
    }
}

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
        kf.inverse_depth =
            filter_depth(studd::two<Image>(observation[1],
                                           Image::Ones(resolution.y(), resolution.x())),
                         kf.intensity);
    }

    studd::two<Image> tracker::filter_depth(studd::two<Image> depth, const Image& intensity)
    {
        auto gradient = sobel(intensity);
        auto g_norm = (gradient[0].cwiseAbs2() + gradient[1].cwiseAbs2()).cwiseSqrt();
        for (int y = 0; y < resolution.y(); y++)
        {
            for (int x = 0; x < resolution.x(); x++)
            {
                if (depth[0](y, x) == 0 ||
                    g_norm(y, x) < 0.2f)
                {
                    depth[1](y, x) = -1;
                }
            }
        }

        return depth;
    }

    Sophus::SE3f tracker::update(const studd::two<Image>& observation, const Sophus::SE3f& guess)
    {
        constexpr float keyframe_distance = 0.1;
        constexpr float keyframe_angle = 2 * M_PI / 32;

        auto& new_intensity = observation[0];
        auto new_depth = studd::two<Image>(observation[1],
                                           Image::Constant(resolution.y(), resolution.x(), 1.0));

        //show_rainbow("before", kf.inverse_depth, observation[0]);
        new_depth = filter_depth(new_depth, new_intensity);
        //show_rainbow("new_depth", new_depth, observation[0]);
        auto sparse_depth = sparsify_depth(kf.inverse_depth);

        //play(sparse_depth, new_intensity, kf.intensity, intrinsic);

        auto current_to_kf_guess = kf.pose.inverse() * pose;
        Sophus::SE3f transform = photometric_tracking<Sophus::SE3Group>(
            sparse_depth,
            new_intensity, kf.intensity, new_depth[0],
            intrinsic,
            weighting,
            current_to_kf_guess,
            16);
        show_residuals("minimized", intrinsic, new_intensity, kf.intensity, sparse_depth,
                       transform, resolution.y(), resolution.x(), 2);
        show_residuals("starting_point", intrinsic, new_intensity, kf.intensity, sparse_depth,
                       current_to_kf_guess, resolution.y(), resolution.x(), 2);
        pose = transform * kf.pose;
        std::cout << "pose:" << std::endl << pose.matrix() << std::endl << guess.matrix() << std::endl;
        auto diff = guess.inverse() * pose;
        std::cout << "tracked error d: " << diff.translation().norm()
                  << ", theta: " << Eigen::AngleAxisf(diff.rotationMatrix()).angle() * 180.0f / M_PI << std::endl;
        std::cout << diff.matrix() << std::endl;
        auto diff2 = kf.pose.inverse() * guess;
        std::cout << "true d: " << diff2.translation().norm()
                  << ", theta: " << Eigen::AngleAxisf(diff2.rotationMatrix()).angle() * 180.0f / M_PI << std::endl;
        std::cout << diff2.matrix() << std::endl;
        auto warped_depth = /*regularize_depth*/(densify_depth(warp(sparsify_depth(new_depth), intrinsic,
                                                                transform.inverse()),
                                             resolution.y(), resolution.x()));
        show_rainbow("warped_depth", warped_depth, observation[0]);
        kf.inverse_depth = fuse_depth(kf.inverse_depth, warped_depth);
        show_rainbow("fused", kf.inverse_depth, observation[0]);
        cv::waitKey(0);

        if ((transform.translation().norm() > keyframe_distance ||
            Eigen::AngleAxisf(transform.rotationMatrix()).angle() > keyframe_angle))
        {
            std::cout << transform.translation() << std::endl;
            std::cout << guess.translation() << std::endl;
            std::cout << "new keyframe" << std::endl;
            // initialize keyframe
            //kf.inverse_depth = /*regularize_depth*/(densify_depth(warp(sparsify_depth(kf.inverse_depth, 1),
            //                                                       intrinsic, transform),
            //                                                  resolution.y(), resolution.x()));
            kf.inverse_depth = new_depth;
            kf.pose = pose;
            kf.intensity = observation[0];
        }
        else
        {
            // TODO: this might need some warping OR just reversing new/ref
            ////auto t_stereo = temporal_stereo(observation[0], kf.left, gradient, transform);
            ////show_rainbow("t_stereo", t_stereo, observation[0]);
            ////kf.inverse_depth = fuse_depth(kf.inverse_depth, t_stereo);
            ////show_rainbow("fused_temporal_static", kf.inverse_depth, observation[0]);
            ////cv::waitKey(0);
            //cv::destroyWindow("fused_temporal_static");
            //kf.inverse_depth = regularize_depth(kf.inverse_depth);
            //show_rainbow("regularized", kf.inverse_depth, observation[0]);
            //cv::waitKey(0);
        }
        //play(observation[0], observation[1]);

        return transform;
    }
}

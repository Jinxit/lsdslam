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
                     const Eigen::Matrix3f& intrinsic,
                     const studd::two<Image>& observation)
        : base_tracker(pose, resolution), intrinsic(intrinsic)
    {
        kf.intensity = observation[0];
        kf.inverse_depth =
            filter_depth(studd::two<Image>(observation[1],
                                           Image::Constant(resolution.y(), resolution.x(), 0.5)),
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
        constexpr float keyframe_distance = 0.2;
        constexpr float keyframe_angle = 2 * M_PI / 16;

        auto& new_intensity = observation[0];
        auto new_depth = studd::two<Image>(observation[1],
                                           Image::Constant(resolution.y(), resolution.x(), 0.5));

        //show_rainbow("before", kf.inverse_depth, observation[0]);
        new_depth = filter_depth(new_depth, new_intensity);
        //show_rainbow("new_depth", new_depth, observation[0]);
        auto sparse_depth = sparsify_depth(kf.inverse_depth);

        //play(sparse_depth, new_intensity, kf.intensity, intrinsic);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 8;
        //options.minimizer_type = ceres::LINE_SEARCH;
        //options.max_lbfgs_rank = 20;
        //options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = 50;
        //options.preconditioner_type = ceres::JACOBI;
        //options.max_num_line_search_step_size_iterations = 10;
        //options.max_solver_time_in_seconds = 0.25;

        float usable_ratio = 0;
        auto current_to_kf_guess = kf.pose.inverse() * pose;
        auto true_transform = kf.pose.inverse() * guess;
        Sophus::SE3f transform = photometric_tracking<Sophus::SE3Group>(
            sparse_depth,
            new_intensity, kf.intensity, kf.inverse_depth[1], new_depth[0],
            intrinsic,
            ceres::HuberLoss(1.0),
            options,
            current_to_kf_guess.inverse(),
            8,
            usable_ratio);
        //Sophus::SE3f transform = true_transform;
        //std::cout << "usable_ratio: " << usable_ratio << std::endl;
        //if (usable_ratio < 1 || transform.translation().norm() > 0.1)
        //    transform = Sophus::SE3f();
        show_residuals("minimized", intrinsic, new_intensity, kf.intensity, sparse_depth,
                       transform.inverse(), resolution.y(), resolution.x(), 2);
        show_residuals("starting_point", intrinsic, new_intensity, kf.intensity, sparse_depth,
                       current_to_kf_guess.inverse(), resolution.y(), resolution.x(), 2);
        show_residuals("groundtruth", intrinsic, new_intensity, kf.intensity, sparse_depth,
                       true_transform.inverse(), resolution.y(), resolution.x(), 2);
        pose = kf.pose * transform.inverse();
        auto warped_depth = /*regularize_depth*/(densify_depth(warp(sparsify_depth(new_depth), intrinsic,
                                                                transform.inverse()),
                                             resolution.y(), resolution.x()));
        show_rainbow("warped_depth", warped_depth, observation[0]);
        kf.inverse_depth = fuse_depth(kf.inverse_depth, warped_depth);
        //cv::waitKey(1);

        if ((transform.translation().norm() > keyframe_distance ||
            Eigen::AngleAxisf(transform.rotationMatrix()).angle() > keyframe_angle))
        {
            //std::cout << transform.translation() << std::endl;
            //std::cout << guess.translation() << std::endl;
            std::cout << "new keyframe" << std::endl;
            std::cout << transform.translation() << std::endl;
            // initialize keyframe
            //kf.inverse_depth = /*regularize_depth*/(densify_depth(warp(sparsify_depth(kf.inverse_depth, 1),
            //                                                       intrinsic, transform),
            //                                                  resolution.y(), resolution.x()));
            kf.inverse_depth = new_depth;
            kf.pose = pose;
            kf.intensity = observation[0];
        }
        else if (false)
        {
            // TODO: this might need some warping OR just reversing new/ref

            // TODO: store in each KF or at least for current
            auto gradient = sobel(kf.intensity);
            auto t_stereo = temporal_stereo(observation[0], kf.intensity,
                                            gradient, true_transform, intrinsic);
            //show_rainbow("t_stereo", t_stereo, observation[0]);
            //show_rainbow("fused_temporal_static", kf.inverse_depth, observation[0]);
            //cv::waitKey(0);
            show_rainbow("temporal_stereo", t_stereo, observation[0]);
            show_rainbow("true_depth", new_depth, observation[0]);
            kf.inverse_depth = fuse_depth(kf.inverse_depth, t_stereo);
            show_rainbow("fused", kf.inverse_depth, observation[0]);
            cv::waitKey(0);
        }
        show_rainbow("fused", kf.inverse_depth, observation[0]);
        //cv::waitKey(0);
        //play(observation[0], observation[1]);

        return transform;
    }
}

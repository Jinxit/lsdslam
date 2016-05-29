#include "stereo.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "../gaussian.hpp"
#include "../two.hpp"
#include "../photometric_tracking.hpp"
#include "../square.hpp"
#include "../misc_utils.hpp"


namespace stereo
{
    tracker::tracker(const Eigen::Affine3f& pose,
                     const Eigen::Vector2i& resolution,
                     const std::function<float(float)>& weighting,
                     const Eigen::Matrix3f& static_fundamental,
                     const Eigen::Affine3f& static_transform,
                     const Eigen::Matrix3f& left_intrinsic,
                     const Eigen::Affine3f& left_transform)
        : base_tracker(pose, resolution, weighting),
          static_fundamental(static_fundamental),
          static_transform(static_transform),
          left_intrinsic(left_intrinsic),
          left_transform(left_transform),
          stereo_epilines(generate_epilines(resolution.y(), resolution.x(), static_fundamental)),
          stereo_epipole(generate_epipole(static_transform, left_intrinsic))
    {
    }

    Eigen::Affine3f tracker::update(const studd::two<Image>& observation, const Eigen::Affine3f& guess)
    {
        constexpr float keyframe_distance = 0.2;
        constexpr float keyframe_angle = M_PI / 8;

        auto& new_left = observation[0];
        auto& new_right = observation[1];

        show_rainbow("first", kf.inverse_depth, new_left);
        auto gradient = sobel(new_left);
        auto s_stereo = regularize_depth(static_stereo(new_left, new_right, gradient));
        show_rainbow("s_stereo", s_stereo, new_left);
        auto sparse_s_stereo = sparsify_depth(s_stereo);
        
        //auto transform = photometric_tracking(sparse_s_stereo,
        //                                      new_left, kf.left,
        //                                      sc.left.intrinsic,
        //                                      weighting,
        //                                      guess).exp();
        auto transform = guess;
        pose = transform * pose;
        Eigen::Affine3f system_change(Eigen::Matrix4f::Zero());
        system_change(0, 1) = 1;
        system_change(1, 0) = -1;
        system_change(2, 2) = 1;
        system_change(3, 3) = 1;
        auto to_left = left_transform * system_change;
        auto kf_to_current = (kf.pose).inverse() * pose * to_left;
        auto warped_s_stereo = regularize_depth(densify_depth(warp(sparse_s_stereo, left_intrinsic,
                                                                   kf_to_current),
                                                resolution.y(), resolution.x()));
        show_rainbow("warped_s_stereo", warped_s_stereo, new_left);
        kf.inverse_depth = fuse_depth(kf.inverse_depth, warped_s_stereo);
        show_rainbow("fused_static", kf.inverse_depth, new_left);

        if (false && (kf_to_current.translation().norm() > keyframe_distance ||
            Eigen::AngleAxisf(kf_to_current.rotation()).angle() > keyframe_angle))
        {
            std::cout << transform.translation() << std::endl;
            std::cout << guess.translation() << std::endl;
            std::cout << "new keyframe" << std::endl;
            // initialize keyframe
            kf.inverse_depth = regularize_depth(densify_depth(warp(sparsify_depth(kf.inverse_depth),
                                                                   left_intrinsic, kf_to_current),
                                                              resolution.y(), resolution.x()));
            kf.pose = pose;
            kf.intensity = new_left;
        }
        else
        {
            // TODO: this might need some warping OR just reversing new/ref
            //auto t_stereo = temporal_stereo(new_left, kf.left, gradient, transform);
            //show_rainbow("t_stereo", t_stereo, new_left);
            //kf.inverse_depth = fuse_depth(kf.inverse_depth, t_stereo);
            //show_rainbow("fused_temporal_static", kf.inverse_depth, new_left);
            //cv::waitKey(0);
            //cv::destroyWindow("fused_temporal_static");
            kf.inverse_depth = regularize_depth(kf.inverse_depth);
            show_rainbow("regularized", kf.inverse_depth, new_left);
            cv::waitKey(0);
        }
        //play(new_left, new_right);

        return transform;
    }

    studd::two<Image> tracker::static_stereo(const Image& left, const Image& right,
                                             const studd::two<Image>& gradient)
    {
        int height = resolution.y();
        int width = resolution.x();

        auto disparity = disparity_rectified(right, left, gradient);

        float bf = ((static_transform.translation().norm()) * left_intrinsic(0, 0));

        size_t j = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                auto disp = Eigen::Vector2f{disparity[0](y, x), 0};
                // TODO: this shit seems a bit strange since it's rectified
                auto geo = geometric_disparity(gradient, stereo_epilines[j], x, y, stereo_epipole);
                auto photo = photometric_disparity(right, left, gradient,
                                                   stereo_epilines[j], x, y, disp, stereo_epipole);

                if (disparity[1](y, x) < 0 || std::min(geo.variance, photo.variance) > 0.01)
                {
                    disparity[0](y, x) = 0;
                    disparity[1](y, x) = -1;
                }
                else
                {
                    // TODO: should this be done before photometric disparity?
                    disparity[0](y, x) = std::abs(disparity[0](y, x)) / bf;
                    // TODO: times alpha^2, figure out what alpha is
                    disparity[1](y, x) = geo.variance + photo.variance;
                }
                j++;
            }
        }
        //show_rainbow("static_stereo", disparity, left);
        //cv::waitKey(0);

        return disparity;
    }
    
}

#include "depth.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "../gaussian.hpp"
#include "../se3.hpp"
#include "../two.hpp"
#include "../photometric_tracking.hpp"
#include "../square.hpp"
#include "../misc_utils.hpp"


inline void play(const sparse_gaussian& sparse_inverse_depth,
                 const Image& new_image, const Image& ref_image,
                 const Eigen::Matrix3f& intrinsic, int pyr = 1)
{
    int height = new_image.rows();
    int width = new_image.cols();
    se3<float> tf;

    while (true)
    {
        int key = cvWaitKey(0);
        if (key == 1113938) // up
        {
            tf.nu.y() += 0.001;
        }
        if (key == 1113940) // down
        {
            tf.nu.y() -= 0.001;
        }
        if (key == 1113937) // left
        {
            tf.nu.x() += 0.001;
        }
        if (key == 1113939) // right
        {
            tf.nu.x() -= 0.001;
        }
        if (key == 1048689) // Q
        {
            tf.omega.x() += 0.001;
        }
        if (key == 1048673) // A
        {
            tf.omega.x() -= 0.001;
        }
        if (key == 1048695) // W
        {
            tf.omega.y() += 0.001;
        }
        if (key == 1048691) // S
        {
            tf.omega.y() -= 0.001;
        }
        if (key == 1048677) // E
        {
            tf.omega.z() += 0.001;
        }
        if (key == 1048676) // D
        {
            tf.omega.z() -= 0.001;
        }
        //std::cout << tf << std::endl;

        auto warped = warp(sparse_inverse_depth, intrinsic, tf.exp());

        Image warped_image = Image::Zero(height, width);
        Image warped_image_inv_depth = Image::Zero(height, width);
        Image warped_image_variance = -Image::Ones(height / pyr, width / pyr);
        Image mask = Image::Zero(height, width);
        for (size_t j = 0; j < warped.size(); j++)
        {
            float x = float(warped[j].first.x()) / pyr;
            float y = float(warped[j].first.y()) / pyr;
            Eigen::Vector2f p(x, y);

            if (x > 0 && x < width / pyr - 1 && y > 0 && y < height / pyr - 1)
            {
                int sx = sparse_inverse_depth[j].first.x() / pyr;
                int sy = sparse_inverse_depth[j].first.y() / pyr;

                if (sx > 0 && sx < width / pyr - 1 && sy > 0 && sy < height / pyr - 1)
                {
                    gaussian warped_pixel = warped[j].second
                                          ^ gaussian(interpolate(warped_image_inv_depth, p),
                                                     interpolate(warped_image_variance, p));
                    if (warped[j].second.mean > warped_image_inv_depth(sy, sx))
                    {
                        if (x * pyr < width && y * pyr < height)
                            warped_image(sy, sx) = interpolate(new_image, p * pyr);
                    }
                    warped_image_inv_depth(sy, sx) = warped_pixel.mean;
                    warped_image_variance(sy, sx) = warped_pixel.variance;
                    mask(sy, sx) = 1;
                }
            }
        }

        show("warped", warped_image);

        show_residuals("warped_residual", intrinsic, new_image, ref_image,
                       sparse_inverse_depth, tf.exp(), height, width, 4);
    }
}


namespace depth
{
    tracker::tracker(const Eigen::Affine3f& pose,
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

    Eigen::Affine3f tracker::update(const studd::two<Image>& observation, const Eigen::Affine3f& guess)
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
        auto transform = ceres_tracking(sparse_depth,
                                              new_intensity, kf.intensity,
                                              intrinsic,
                                              weighting,
                                              current_to_kf_before,
                                              4).exp();
        show_residuals("minimized", intrinsic, new_intensity, kf.intensity, sparse_depth,
                       transform, resolution.y(), resolution.x(), 2);
        std::cout << "debug: " << std::endl;
        std::cout << current_to_kf_before.matrix() << std::endl << std::endl;
        std::cout << transform.matrix() << std::endl;
        transform = guess;
        pose = pose * transform;
        Eigen::Affine3f current_to_kf = kf.pose.inverse() * pose;
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

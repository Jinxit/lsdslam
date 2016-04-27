#include "tracker.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "gaussian.hpp"
#include "se3.hpp"
#include "two.hpp"
#include "photometric_tracking.hpp"
#include "square.hpp"
#include "misc_utils.hpp"

namespace
{
    studd::two<Image> regularize_depth(const studd::two<Image>& inverse_depth)
    {
        auto height = inverse_depth[0].rows();
        auto width = inverse_depth[0].cols();

        Image output_inv_depth = Image::Zero(height, width);
        Image output_variance = Image::Zero(height, width);

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int num_added = 0;

                auto original_inv_depth = inverse_depth[0](y, x);
                auto original_variance = inverse_depth[1](y, x);
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        auto adjacent_inv_depth = inverse_depth[0](y + dy, x + dx);
                        auto adjacent_variance = inverse_depth[1](y + dy, x + dx);

                        //if (std::abs(adjacent_inv_depth - original_inv_depth)
                        //    > original_variance * 2
                        // || adjacent_variance < 0)
                        //    continue;

                        output_inv_depth(y, x) += adjacent_inv_depth;
                        num_added++;
                    }
                }

                if (num_added > 0)
                {
                    output_inv_depth(y, x) /= num_added;
                    output_variance(y, x) = original_variance;
                }
                else
                {
                    output_inv_depth(y, x) = 0;
                    output_variance(y, x) = -1;
                }
            }
        }

        return {output_inv_depth, output_variance};
    }

    studd::two<Image> fuse_depth(const studd::two<Image>& lhs, const studd::two<Image>& rhs)
    {
        studd::two<Image> output = studd::make_two(Image(lhs[0].rows(), lhs[0].cols()),
                                                   Image(lhs[0].rows(), lhs[0].cols()));
        for (int y = 0; y < lhs[0].rows(); y++)
        {
            for (int x = 0; x < lhs[0].cols(); x++)
            {
                auto fused = gaussian(lhs[0](y, x), lhs[1](y, x))
                           ^ gaussian(rhs[0](y, x), rhs[1](y, x));
                output[0](y, x) = fused.mean;
                output[1](y, x) = fused.variance;
            }
        }
        return output;
    }

    Eigen::Matrix3f fundamental_from_transform(const Eigen::Affine3f& transform,
                                               const Eigen::Matrix3f& intrinsic)
    {
        Eigen::Vector3f e = intrinsic * transform.rotation().transpose() * transform.translation();
        return intrinsic.transpose().inverse() * transform.rotation() * skewed(e);
    }
}

tracker::tracker(const euroc::stereo_calibration& sc, const Eigen::Affine3f& pose,
                 const Image& new_left, const Image& new_right)
    : pose(pose),
      sc(sc),
      kf(sc.resolution.y(), sc.resolution.x()),
      stereo_epilines(generate_epilines(sc.resolution.y(), sc.resolution.x(),
                                        sc.static_fundamental)),
      stereo_epipole(generate_epipole(sc.transform_left_right, sc.left.intrinsic)),
      weighting([](float r) { return 1; })
{
    auto gradient = sobel(new_left);
    kf.inverse_depth = regularize_depth(static_stereo(new_left, new_right, gradient));
    kf.left = new_left;
    kf.right = new_right;
    kf.pose = pose;
}

void tracker::play(const Image& new_left, const Image& new_right)
{
    int height = sc.resolution.y();
    int width = sc.resolution.x();
    se3 tf;
    auto gradient = sobel(new_left);
    auto s_stereo = static_stereo(new_left, new_right, gradient);
    kf.inverse_depth = s_stereo;//regularize_depth(s_stereo);
    auto sparse_depth = sparsify_depth(kf.inverse_depth);

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
            tf.omega.x() += 0.01;
        }
        if (key == 1048673) // A
        {
            tf.omega.x() -= 0.01;
        }
        if (key == 1048695) // W
        {
            tf.omega.y() += 0.01;
        }
        if (key == 1048691) // S
        {
            tf.omega.y() -= 0.01;
        }
        if (key == 1048677) // E
        {
            tf.omega.z() += 0.01;
        }
        if (key == 1048676) // D
        {
            tf.omega.z() -= 0.01;
        }
        std::cout << key << std::endl;
        std::cout << tf << std::endl;

        auto warped = warp(sparse_depth[0], sc.left.intrinsic, tf.exp());

        Image warped_image = Image::Zero(height, width);
        Image warped_image_inv_depth = Image::Zero(height, width);
        for (size_t i = 0; i < warped.size(); i++)
        {
            int x = warped[i].first.x();
            int y = warped[i].first.y();
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                int sx = sparse_depth[0][i].first.x();
                int sy = sparse_depth[0][i].first.y();

                if (warped_image_inv_depth(y, x) < warped[i].second.mean)
                {
                    warped_image_inv_depth(y, x) = warped[i].second.mean;
                    warped_image(y, x) = new_left(sy, sx);
                }
                else
                {
                    warped_image_inv_depth(y, x) = 0;
                }
            }
        }

        //show("warped", warped_image);
        show_rainbow("warped", densify_depth(warped, height, width), new_left);
    }
}


Eigen::Affine3f tracker::update(const Image& new_left, const Image& new_right, const Eigen::Affine3f& guess)
{
    constexpr float keyframe_distance = 0.2;
    constexpr float keyframe_angle = M_PI / 8;

    show_rainbow("first", kf.inverse_depth, new_left);
    auto gradient = sobel(new_left);
    auto s_stereo = regularize_depth(static_stereo(new_left, new_right, gradient));
    show_rainbow("s_stereo", s_stereo, new_left);
    auto sparse_s_stereo = sparsify_depth(s_stereo, 1)[0];
    
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
    auto to_left = sc.transform_left * system_change;
    auto kf_to_current = (kf.pose).inverse() * pose * to_left;
    auto warped_s_stereo = regularize_depth(densify_depth(warp(sparse_s_stereo, sc.left.intrinsic,
                                              kf_to_current),
                                         sc.resolution.y(), sc.resolution.x()));
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
        kf.inverse_depth = regularize_depth(densify_depth(warp(sparsify_depth(kf.inverse_depth, 1)[0],
                                                               sc.left.intrinsic, kf_to_current),
                                                          sc.resolution.y(), sc.resolution.x()));
        kf.pose = pose;
        kf.left = new_left;
        kf.right = new_right;
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
    int height = sc.resolution.y();
    int width = sc.resolution.x();

    auto disparity = disparity_rectified(right, left, gradient);

    float bf = ((sc.transform_left_right.translation().norm()) * sc.left.intrinsic(0, 0));

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


studd::two<Image> tracker::temporal_stereo(const Image& new_image, const Image& ref_image,
                                           const studd::two<Image>& gradient,
                                           const Eigen::Affine3f& transform)
{
    int height = sc.resolution.y();
    int width = sc.resolution.x();

    auto fundamental = fundamental_from_transform(transform, sc.left.intrinsic);
    auto epilines = generate_epilines(height, width, fundamental);
    auto epipole = generate_epipole(transform, sc.left.intrinsic);
    auto disparity = disparity_epilines(new_image, ref_image, epilines, epipole);

    // TODO: investigate, transform might not equal baseline
    float bf = ((transform.translation().norm()) * sc.left.intrinsic(1, 1));

    size_t j = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            auto disp = Eigen::Vector2f{disparity[0](y, x), 0};
            // TODO: this shit seems a bit strange since it's rectified
            // also why don't we use the mean here?
            auto geo = geometric_disparity(gradient, epilines[j], x, y, epipole);
            auto photo = photometric_disparity(new_image, ref_image, gradient,
                                               epilines[j], x, y, disp, epipole);

            if (std::min(geo.variance, photo.variance) > 0.01)
            {
                disparity[0](y, x) = 0;
                disparity[1](y, x) = -1;
            }
            else
            {
                // TODO: should this be done before photometric disparity?
                disparity[0](y, x) /= bf;
                // TODO: times alpha^2, figure out what alpha is
                disparity[1](y, x) = geo.variance + photo.variance;
            }
            j++;
        }
    }

    return disparity;
}
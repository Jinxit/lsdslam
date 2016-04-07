#include "tracker.hpp"

#include "gaussian.hpp"
#include "se3.hpp"
#include "two.hpp"
#include "photometric_tracking.hpp"
#include "square.hpp"

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

                        if (std::abs(adjacent_inv_depth - original_inv_depth)
                            > original_variance * 2
                         || adjacent_variance < 0)
                            continue;

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

    std::vector<std::vector<pixel>> sparsify_depth(const studd::two<Image>& inverse_depth)
    {
        constexpr int max_pyramid = 4;

        auto height = inverse_depth[0].rows();
        auto width = inverse_depth[0].cols();

        std::vector<std::vector<pixel>> sparse_inverse_depth;
        //sparse_inverse_depth.reserve(height * width);
        for (int pyr = max_pyramid; pyr > 0; pyr /= 2)
        {
            sparse_inverse_depth.emplace_back();
            for (int y = 0; y < height; y += pyr)
            {
                for (int x = 0; x < width; x += pyr)
                {
                    auto inv_mean = inverse_depth[0](y, x);
                    auto inv_var = inverse_depth[1](y, x);

                    if (inv_var >= 0)
                    {
                        sparse_inverse_depth.back().emplace_back(inv_mean, inv_var, x, y);
                    }
                }
            }
        }

        return sparse_inverse_depth;
    }

    Eigen::Matrix3f fundamental_from_transform(const Eigen::Affine3f& transform,
                                               const Eigen::Matrix3f& intrinsic)
    {
        Eigen::Vector3f e = intrinsic * transform.rotation().transpose() * transform.translation();
        return intrinsic.transpose().inverse() * transform.rotation() * skewed(e);
    }
}

tracker::tracker(const stereo_calibration& sc)
    : sc(sc),
      kf(sc.resolution.y(), sc.resolution.x()),
      stereo_epilines(generate_epilines(sc.resolution.y(), sc.resolution.x(),
                                        sc.static_fundamental)),
      stereo_epipole(generate_epipole(sc.transform, sc.left.intrinsic)),
      weighting([](float r) { return 1; })
{ }


Eigen::Affine3f tracker::update(const Image& new_left, const Image& new_right)
{
    constexpr float keyframe_distance = 0.2;

    auto gradient = sobel(new_left);
    auto s_stereo = static_stereo(new_left, new_right, gradient);
    kf.inverse_depth = fuse_depth(kf.inverse_depth, s_stereo);
    
    auto transform = photometric_tracking(sparsify_depth(kf.inverse_depth),
                                          new_left, kf.left,
                                          sc.left.intrinsic,
                                          weighting);

    if (transform.translation().norm() > keyframe_distance)
    {
        // initialize keyframe
        kf.inverse_depth = regularize_depth(kf.inverse_depth);
        kf.position = position;
        kf.left = new_left;
        kf.right = new_right;
    }
    else
    {
        // TODO: this might need some warping OR just reversing new/ref
        auto t_stereo = temporal_stereo(new_left, kf.left, gradient, transform);
        kf.inverse_depth = fuse_depth(kf.inverse_depth, t_stereo);
        kf.inverse_depth = regularize_depth(kf.inverse_depth);
    }

    return transform;
}

studd::two<Image> tracker::static_stereo(const Image& left, const Image& right,
                                         const studd::two<Image>& gradient)
{
    int height = sc.resolution.y();
    int width = sc.resolution.x();

    auto disparity = disparity_rectified(right, left, gradient);

    float bf = ((sc.transform.translation().norm()) * sc.left.intrinsic(1, 1));

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
                disparity[0](y, x) /= bf;
                // TODO: times alpha^2, figure out what alpha is
                disparity[1](y, x) = geo.variance + photo.variance;
            }
            j++;
        }
    }

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
#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <sophus/se3.hpp>

#include "../two.hpp"
#include "../eigen_utils.hpp"
#include "../disparity.hpp"
#include "../epiline.hpp"

struct keyframe
{
    keyframe(int height, int width, const Sophus::SE3f& pose)
        : intensity(Image::Constant(height, width, 0)),
          inverse_depth(Image::Constant(height, width, 0),
                        Image::Constant(height, width, -1)),
          pose(pose)
    { }

    Image intensity;
    studd::two<Image> inverse_depth;
    Sophus::SE3f pose;
};

template<class Observation>
class base_tracker
{
public:
    base_tracker(const Sophus::SE3f& pose, const Eigen::Vector2i& resolution,
                 const std::function<float(float)>& weighting)
        : pose(pose),
          resolution(resolution),
          kf(resolution.y(), resolution.x(), pose),
          weighting(weighting)
    { }

    virtual Sophus::SE3f update(const Observation& o, const Sophus::SE3f& guess) = 0;
    Sophus::SE3f get_pose() const { return pose; }

protected:
    studd::two<Image> temporal_stereo(const Image& new_image, const Image& ref_image,
                                      const studd::two<Image>& gradient,
                                      const Sophus::SE3f& transform,
                                      const Eigen::Matrix3f& intrinsic)
    {
        // TODO: this is def not working
        /*int height = resolution.y();
        int width = resolution.x();

        auto fundamental = fundamental_from_transform(transform, intrinsic);
        auto epilines = generate_epilines(height, width, fundamental);
        auto epipole = generate_epipole(transform, intrinsic);
        auto disparity = disparity_epilines(new_image, ref_image, epilines, epipole);

        // TODO: investigate, transform might not equal baseline
        float bf = ((transform.translation().norm()) * intrinsic(1, 1));

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

        return disparity;*/
        return studd::two<Image>();
    }

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

                //if (original_variance < 0)
                //    continue;

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
                        //if (adjacent_variance < 0)
                        //    continue;

                        output_inv_depth(y, x) += adjacent_inv_depth;
                        num_added++;
                    }
                }

                if (num_added > 1)
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

    Eigen::Matrix3f fundamental_from_transform(const Sophus::SE3f& transform,
                                               const Eigen::Matrix3f& intrinsic)
    {
        Eigen::Vector3f e = intrinsic * transform.rotationMatrix().transpose()
                                      * transform.translation();
        return intrinsic.transpose().inverse() * transform.rotationMatrix() * skewed(e);
    }

    Sophus::SE3f pose;
    Eigen::Vector2i resolution;
    keyframe kf;
    std::function<float(float)> weighting;
};
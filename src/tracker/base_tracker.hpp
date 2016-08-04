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
    base_tracker(const Sophus::SE3f& pose, const Eigen::Vector2i& resolution)
        : pose(pose),
          resolution(resolution),
          kf(resolution.y(), resolution.x(), pose)
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
        int height = resolution.y();
        int width = resolution.x();

        auto fundamental = fundamental_from_transform(transform, intrinsic);
        auto epilines = generate_epilines(height, width, fundamental);
        auto epipole = generate_epipole(transform, intrinsic);
        auto disparity = disparity_epilines(new_image, ref_image, epilines, epipole);
        Image geo_im = Image::Zero(height, width);
        Image photo_im = Image::Zero(height, width);
        show_epilines("epilines", epilines, epipole, new_image);
        auto uh = (disparity[0].cwiseProduct(disparity[0]) + disparity[1].cwiseProduct(disparity[1])).cwiseSqrt().eval();
        show("xy_disp", uh, true);

        Eigen::Matrix<float, 3, 4> Ma = intrinsic * Sophus::SE3f().matrix3x4();
        Eigen::Matrix<float, 3, 4> Mb = intrinsic * transform.matrix3x4();
        auto depth = [&](float xa, float ya, float xb, float yb) {
            Eigen::MatrixXf A = Eigen::Matrix4f();
            A.row(0) = xa * (Ma.row(2) - Ma.row(0));
            A.row(1) = ya * (Ma.row(2) - Ma.row(1));
            A.row(2) = xb * (Mb.row(2) - Mb.row(0));
            A.row(3) = yb * (Mb.row(2) - Mb.row(1));

            Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::Matrix<float, 4, 1> coords = svd.singularValues();

            if (coords(3) == 0)
            {
                return 0.0f;
            }
            return coords(2) / coords(3);
        };

        size_t j = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                auto dx = disparity[0](y, x);
                auto dy = disparity[1](y, x);

                if (dx != dx || dy != dy)
                {
                    disparity[0](y, x) = 0;
                    disparity[1](y, x) = -1;
                }

                auto disp = Eigen::Vector2f{dx, dy};
                // also why don't we use the mean here?
                auto geo = geometric_disparity(gradient, epilines[j], x, y, epipole);
                auto photo = photometric_disparity(new_image, ref_image, gradient,
                                                   epilines[j], x, y, disp, epipole);
                geo_im(y, x) = std::log(geo.variance);
                photo_im(y, x) = std::log(photo.variance);

                if (std::min(geo.variance, photo.variance) > 0.05)
                {
                    disparity[0](y, x) = 0;
                    disparity[1](y, x) = -1;
                }
                else
                {
                    // TODO: should this be done before photometric disparity?
                    auto d = depth(x, y, x - dx, y - dy);
                    if (d > 0)
                    {
                        disparity[0](y, x) = d;
                        // TODO: times alpha^2, figure out what alpha is
                        disparity[1](y, x) = geo.variance + photo.variance;
                    }
                    else
                    {
                        disparity[0](y, x) = 0;
                        disparity[1](y, x) = -1;
                    }
                }
                j++;
            }
        }
        show("geo", geo_im, true);
        show("photo", photo_im, true);

        return disparity;
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
                if (lhs[1](y, x) > 0 || true)
                {
                    auto fused = gaussian(lhs[0](y, x), lhs[1](y, x))
                               ^ gaussian(rhs[0](y, x), rhs[1](y, x));
                    output[0](y, x) = fused.mean;
                    output[1](y, x) = fused.variance;
                }
                else
                {
                    output[1](y, x) = -1;
                }
            }
        }
        return output;
    }

    Eigen::Matrix3f fundamental_from_transform(const Sophus::SE3f& transform,
                                               const Eigen::Matrix3f& intrinsic)
    {
        Eigen::Matrix3f ess = skewed(transform.translation()) * transform.rotationMatrix();
        Eigen::Matrix3f fun = intrinsic.transpose().inverse() * ess * intrinsic.inverse();
        return fun;
    }

    Sophus::SE3f pose;
    Eigen::Vector2i resolution;
    keyframe kf;
};
#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

inline void show(const std::string& title, const Image& image, bool normalize = false,
          float multiplier = 1.0f)
{
    cv::Mat cv_image;
    cv::eigen2cv(image, cv_image);
    if (normalize)
        cv::normalize(cv_image, cv_image, 1, 0, cv::NORM_INF);
    //std::cout << "show " << image(0, 0) << " " << cv_image.at<float>(0, 0) << std::endl;
    cv::imshow(title, cv_image * multiplier);
}

inline void show_rainbow(const std::string& title, const studd::two<Image>& image,
                         const Image& base)
{
    auto height = image[0].rows();
    auto width = image[0].cols();

    cv::Mat h = cv::Mat::zeros(height, width, CV_32F);;
    cv::Mat s = cv::Mat::ones(height, width, CV_32F);
    cv::Mat v = cv::Mat::ones(height, width, CV_32F);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (image[1](y, x) < 0 || x < 5 || x > width - 5 || y < 5 || y > height - 5
                || image[0](y, x) == -1.0f)
            {
                v.at<float>(y, x) = 0;
            }
            else if (image[0](y, x) != 0)
            {
                h.at<float>(y, x) = std::abs(image[0](y, x));
            }
        }
    }
    cv::normalize(h, h, 255, 0, cv::NORM_INF);
    cv::Mat cv_image;
    cv::merge(std::vector<cv::Mat>{h, s, v}, cv_image);
    cv::cvtColor(cv_image, cv_image, CV_HSV2RGB);
    /*for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (image[1](y, x) < 0 || x < 5 || x > width - 5 || y < 5 || y > height - 5)
            {
                cv_image.at<cv::Vec3f>(y, x) = cv::Vec3f(image[0](y, x), image[0](y, x), image[0](y, x));
            }
        }
    }*/
    cv::imshow(title, cv_image);
}

/*inline void plot_errors(const se3& id,
                        const sparse_gaussian& sparse_inverse_depth,
                        const Image& new_image,
                        const Image& ref_image,
                        const Eigen::Matrix3f& intrinsic,
                        const std::function<float(float)>& weighting)
{
    static int file_counter = 0;

    auto height = new_image.rows();
    auto width = new_image.cols();

    auto calc_error = [&](const se3 ksi, bool save = false) {
        auto warped = warp(sparse_inverse_depth, intrinsic, ksi.exp());
        Image warped_image = Image::Zero(height, width);
        Image warped_image_inv_depth = Image::Zero(height, width);
        for (size_t i = 0; i < warped.size(); i++)
        {
            int x = warped[i].x;
            int y = warped[i].y;
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                int sx = sparse_inverse_depth[i].x;
                int sy = sparse_inverse_depth[i].y;

                if (warped_image_inv_depth(y, x) < warped[i].inverse_depth)
                {
                    warped_image_inv_depth(y, x) = warped[i].inverse_depth;
                    warped_image(y, x) = ref_image(sy, sx);
                }
            }
        }

        if (save)
        {
            cv::Mat cv_image;
            cv::eigen2cv(warped_image, cv_image);
            std::stringstream uh;
            uh << "warped/" << std::setfill('0') << std::setw(5) << file_counter << "_" << ksi << ".jpg";
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (cv_image.at<float>(y, x) == 0)
                        cv_image.at<float>(y, x) = std::abs(cv_image.at<float>(y, x) - new_image(y, x));
                }
            }
            cv::imwrite(uh.str(), cv_image * 255);
            file_counter++;
        }
        studd::two<Image> gradient = sobel(warped_image);
        Eigen::VectorXf residuals;
        Eigen::DiagonalMatrix<float, Eigen::Dynamic> weights;
        std::tie(residuals, weights) = residuals_and_weights(sparse_inverse_depth, warped,
                                                             new_image, ref_image, weighting);
        float error = ((residuals.transpose() * weights).dot(residuals)
                    / sparse_inverse_depth.size());
        return error;
    };

    auto lerp = [](float v0, float v1, float t) {
        return (1 - t) * v0 + t * v1;
    };
    for (float t = 0.0f; t <= 1.0f; t += 0.005f)
    {
        calc_error(se3(lerp(0, id.omega[0], t),
                       lerp(0, id.omega[1], t),
                       lerp(0, id.omega[2], t),
                       lerp(0, id.nu[0], t),
                       lerp(0, id.nu[1], t),
                       lerp(0, id.nu[2], t)), true);
    }
    return;

    for (float val = -50.0f; val < 50.0f; val += 5.0f)
    {
        calc_error(id ^ se3(0, 0, 0, val, 0, 0), true);
    }
    for (float val = -50.0f; val < 50.0f; val += 5.0f)
    {
        calc_error(id ^ se3(0, 0, 0, 0, val, 0), true);
    }
    for (float val = -50.0f; val < 50.0f; val += 5.0f)
    {
        calc_error(id ^ se3(0, 0, 0, 0, 0, val), true);
    }
    for (float val = -0.15f; val < 0.15f; val += 0.005f)
    {
        calc_error(id ^ se3(val, 0, 0, 0, 0, 0), true);
    }
    for (float val = -0.15f; val < 0.15f; val += 0.005f)
    {
        calc_error(id ^ se3(0, val, 0, 0, 0, 0), true);
    }
    for (float val = -0.15f; val < 0.15f; val += 0.005f)
    {
        calc_error(id ^ se3(0, 0, val, 0, 0, 0), true);
    }
    exit(0);

    auto start = std::chrono::steady_clock::now();
    //std::cout << "errors" << std::endl;
    int count = 0;
    static std::ofstream f("errors.txt");
    static std::ofstream g("minima.txt");
    static std::ofstream h("starts.txt");
    static int val_count = 0;

    se3 min_tf = id;
    float min_error = calc_error(id);
    se3 tf;
    float error;
    bool found_smaller = false;
    for (float val = -0.4f; val < 0.4f; val += 0.04f)
    {
        std::cout << "warping " << val << std::endl;
        if (std::abs(val) < 0.0001)
        {
            g << val_count << std::endl;
        }

        tf = id ^ se3(0, 0, 0, val * 2, 0, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.nu[0]) < 1e-2)
            h << 0 << " " << val_count << std::endl;

        tf = id ^ se3(0, 0, 0, 0, val * 2, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.nu[1]) < 1e-2)
            h << 1 << " " << val_count << std::endl;

        tf = id ^ se3(0, 0, 0, 0, 0, val * 2);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.nu[2]) < 1e-2)
            h << 2 << " " << val_count << std::endl;

        tf = id ^ se3(val, 0, 0, 0, 0, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.omega[0]) < 1e-2)
            h << 3 << " " << val_count << std::endl;

        tf = id ^ se3(0, val, 0, 0, 0, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.omega[1]) < 1e-2)
            h << 4 << " " << val_count << std::endl;

        tf = id ^ se3(0, 0, val, 0, 0, 0);
        error = calc_error(tf);
        if (error < min_error)
        {
            found_smaller = true;
            min_error = error;
            min_tf = tf;
        }
        f << error << " ";
        if (std::abs(tf.omega[2]) < 1e-2)
            h << 5 << " " << val_count << std::endl;

        f << std::endl;
        count += 6;
        val_count++;
    }
    if (!found_smaller)
    {
        std::cout << "finished search, saving images" << std::endl;
        std::cout << "ksi: " << min_tf << std::endl;
        for (float t = 0.0f; t <= 1.0f; t += 0.005f)
        {
            calc_error(se3(lerp(0, id.omega[0], t),
                           lerp(0, id.omega[1], t),
                           lerp(0, id.omega[2], t),
                           lerp(0, id.nu[0], t),
                           lerp(0, id.nu[1], t),
                           lerp(0, id.nu[2], t)), true);
        }
    }
    else
    {
        std::cout << "found smaller! " << min_error << std::endl;
        return plot_errors(min_tf, sparse_inverse_depth, new_image, ref_image, intrinsic, weighting);
    }
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    //std::cout << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
    //std::cout << "for " << count << " warps" << std::endl;
}
*/
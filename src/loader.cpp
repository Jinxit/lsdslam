#include "loader.hpp"

namespace fs = boost::filesystem;

namespace
{
    std::function<Image(size_t)>
    create_image_loader(const std::string& folder, const calibration& c)
    {
        return [&c, folder](size_t id) {
            cv::Mat image = cv::imread(folder + std::to_string(id) + ".png",
                                       CV_LOAD_IMAGE_GRAYSCALE);

            static int i = 0;
            cv::imwrite("output/rectified/" + std::to_string(i) + "_before.png", image);
            cv::remap(image, image, c.map_x, c.map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

            Image output = Image::Zero(image.rows, image.cols);
            for (int y = 0; y < image.rows; y++)
            {
                for (int x = 0; x < image.cols; x++)
                {
                    output(y, x) = image.at<unsigned char>(y, x) / 255.0;
                }
            }

            cv::imwrite("output/rectified/" + std::to_string(i++) + "_after.png", image);

            return output;
        };
    }

    bool timestamp_search(const std::pair<size_t, Eigen::Affine3f>& pose, size_t i)
    {
        return pose.first < i;
    }

    std::vector<size_t> load_indices(const std::string& folder)
    {
        std::vector<size_t> output;
        fs::path p(folder + "/cam0/data/");
        for(auto& entry : boost::make_iterator_range(fs::directory_iterator(p), {}))
        {
            std::string filename = entry.path().filename().string();
            for (size_t i = 0; i < 4; i++)
            {
                filename.pop_back();
            }
            output.push_back(std::stol(filename));
        }
        return output;
    }

    std::vector<std::pair<size_t, Eigen::Affine3f>> load_poses(const std::string& folder)
    {
        std::ifstream pose_stream(folder + "/state_groundtruth_estimate0/data.csv");

        std::string line;
        std::vector<std::pair<size_t, Eigen::Affine3f>> poses;

        // eat the first line, hacky
        std::getline(pose_stream, line);
        while (std::getline(pose_stream, line))
        {
            std::stringstream line_stream(line);
            std::string cell;

            size_t timestamp;
            std::getline(line_stream, cell, ',');
            timestamp = std::stol(cell);

            Eigen::Translation<float, 3> pos;
            std::getline(line_stream, cell, ',');
            pos.x() = std::stof(cell);
            std::getline(line_stream, cell, ',');
            pos.y() = std::stof(cell);
            std::getline(line_stream, cell, ',');
            pos.z() = std::stof(cell);

            Eigen::Quaternionf rot;
            std::getline(line_stream, cell, ',');
            rot.w() = std::stof(cell);
            std::getline(line_stream, cell, ',');
            rot.x() = std::stof(cell);
            std::getline(line_stream, cell, ',');
            rot.y() = std::stof(cell);
            std::getline(line_stream, cell, ',');
            rot.z() = std::stof(cell);

            Eigen::Affine3f pose;
            pose = rot;
            pose.translation() = pos.translation();
            poses.emplace_back(timestamp, pose);
        }
        return poses;
    }

    stereo_calibration load_calibration(const std::string& folder)
    {
        stereo_calibration output;

        // load calibration from file
        YAML::Node left = YAML::LoadFile(folder + "cam0/sensor.yaml");
        YAML::Node right = YAML::LoadFile(folder + "cam1/sensor.yaml");

        for (size_t i = 0; i < 2; i++)
        {
            output.left.distortion.k[i] = left["distortion_coefficients"][i].as<float>();
            output.left.distortion.p[i] = left["distortion_coefficients"][i + 2].as<float>();
            output.right.distortion.k[i] = right["distortion_coefficients"][i].as<float>();
            output.right.distortion.p[i] = right["distortion_coefficients"][i + 2].as<float>();
        }

        for (size_t i = 0; i < 4 * 4; i++)
        {
            output.left.extrinsic(i / 4, i % 4) = left["T_BS2"]["data"][i].as<float>();
            output.right.extrinsic(i / 4, i % 4) = right["T_BS2"]["data"][i].as<float>();
        }

        cv::Mat left_intrinsic = cv::Mat::zeros(3, 3, CV_32F);
        left_intrinsic.at<float>(0, 0) = left["intrinsics"][0].as<float>();
        left_intrinsic.at<float>(1, 1) = left["intrinsics"][1].as<float>();
        left_intrinsic.at<float>(0, 2) = left["intrinsics"][2].as<float>();
        left_intrinsic.at<float>(1, 2) = left["intrinsics"][3].as<float>();
        left_intrinsic.at<float>(2, 2) = 1;
        cv::Mat right_intrinsic = cv::Mat::zeros(3, 3, CV_32F);
        right_intrinsic.at<float>(0, 0) = right["intrinsics"][0].as<float>();
        right_intrinsic.at<float>(1, 1) = right["intrinsics"][1].as<float>();
        right_intrinsic.at<float>(0, 2) = right["intrinsics"][2].as<float>();
        right_intrinsic.at<float>(1, 2) = right["intrinsics"][3].as<float>();
        right_intrinsic.at<float>(2, 2) = 1;

        // set up rectification and undistortion maps
        int width = left["resolution"][0].as<int>();
        int height = left["resolution"][1].as<int>();

        Eigen::Affine3f T = output.left.extrinsic.inverse() * output.right.extrinsic;
        cv::Mat R_cv = cv::Mat::zeros(3, 3, CV_64F);
        cv::eigen2cv(T.rotation(), R_cv);
        R_cv.convertTo(R_cv, CV_64F);
        cv::Mat T_cv = cv::Mat::zeros(3, 1, CV_64F);
        cv::eigen2cv(T.translation().matrix().eval(), T_cv);
        T_cv.convertTo(T_cv, CV_64F);

        cv::Mat left_rectification, right_rectification, left_perspective, right_perspective;
        cv::stereoRectify(left_intrinsic, output.left.distortion.as_vector(),
                          right_intrinsic, output.right.distortion.as_vector(),
                          cv::Size(width, height), R_cv, T_cv,
                          left_rectification, right_rectification,
                          left_perspective, right_perspective, cv::noArray(),
        //                  CV_CALIB_ZERO_DISPARITY, 0);
                          0, 0);

        cv::initUndistortRectifyMap(left_intrinsic, output.left.distortion.as_vector(),
                                    left_rectification, left_perspective, cv::Size(width, height),
                                    CV_32FC1, output.left.map_x, output.left.map_y);
        cv::initUndistortRectifyMap(right_intrinsic, output.right.distortion.as_vector(),
                                    right_rectification, right_perspective, cv::Size(width, height),
                                    CV_32FC1, output.right.map_x, output.right.map_y);

        cv::cv2eigen(left_perspective.colRange(0, 3), output.left.intrinsic);
        cv::cv2eigen(right_perspective.colRange(0, 3), output.right.intrinsic);

        // multiple view geometry, table 9.1
        output.static_fundamental = output.right.intrinsic.inverse().transpose()
                                  * T.rotation() * output.left.intrinsic.transpose()
                                  * skewed(output.left.intrinsic * T.rotation().transpose()
                                         * T.translation());

        output.resolution = Eigen::Vector2i(width, height);

        output.transform = T;

        return output;
    }
}

loader::loader(const std::string& folder)
    : folder(folder), c(load_calibration(folder)),
      indices(load_indices(folder)),
      left_map(create_image_loader(folder + "cam0/data/", c.left)),
      right_map(create_image_loader(folder + "cam1/data/", c.right))
{
    left_map[indices[198]];
    right_map[indices[198]];
    poses = load_poses(folder);
    //recalibrate();
}

stereo_frame loader::operator()(size_t i, size_t j)
{
    stereo_frame output;

    output.before = (*this)[i];
    output.after = (*this)[j];
    output.transform = se3(output.before.pose.inverse() * output.after.pose);

    return output;
}

frame loader::operator[](size_t i)
{
    frame output;
    output.left = left_map[indices[i]];
    output.right = right_map[indices[i]];
    output.pose = pose_at(indices[i]);

    return output;
}

Eigen::Affine3f loader::pose_at(size_t i)
{
    auto it_after = std::lower_bound(poses.begin(), poses.end(),
                                     i, &timestamp_search);
    auto it_before = it_after;
    if (it_after->first == i)
    {
        ++it_after;
    }
    else
    {
        --it_before;
    }

    float t = float(i - it_before->first) / (it_after->first - it_before->first);
    auto interp = interpolate(it_before->second, it_after->second, t);
    return interp;
}

void loader::recalibrate()
{
    constexpr size_t frame_skip = 20;

    YAML::Node info = YAML::LoadFile(folder + "checkerboard_7x6.yaml");
    cv::Size board_size(info["targetCols"].as<size_t>(), info["targetRows"].as<size_t>());

    float col_spacing = info["colSpacingMeters"].as<float>();
    float row_spacing = info["rowSpacingMeters"].as<float>();

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points1, image_points2;
    std::vector<cv::Point2f> corners1, corners2;

    std::vector<cv::Point3f> obj;
    for (int y = 0; y < board_size.width; y++)
    {
        for (int x = 0; x < board_size.height; x++)
        {
            obj.push_back(cv::Point3f(x * col_spacing, y * row_spacing, 0.0f));
        }
    }

    for (size_t i = 200; i <= indices.size(); i += frame_skip)
    {
        std::cout << (float(i) / indices.size()) << std::endl;
        cv::Mat left, right;
        cv::eigen2cv(left_map[indices[i]], left);
        left.convertTo(left, CV_8U, 255);
        cv::eigen2cv(right_map[indices[i]], right);
        right.convertTo(right, CV_8U, 255);

        bool found1 = cv::findChessboardCorners(left, board_size, corners1, 
                                                CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        bool found2 = cv::findChessboardCorners(right, board_size, corners2, 
                                                CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found1 && found2)
        {
            cornerSubPix(left, corners1, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.));
            cornerSubPix(right, corners2, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.));
            //drawChessboardCorners(left, board_size, corners1, found1);
            //imshow("left", left);
            //cv::waitKey(0);

            image_points1.push_back(corners1);
            image_points2.push_back(corners2);
            object_points.push_back(obj);
        }
    }


    cv::Mat R, T, E, F, K1, K2;
    cv::eigen2cv(c.left.intrinsic, K1);
    cv::eigen2cv(c.right.intrinsic, K2);
    cv::Mat D1(c.left.distortion.as_vector(), true);
    cv::Mat D2(c.left.distortion.as_vector(), true);
    cv::stereoCalibrate(object_points, image_points1, image_points2, 
                        K1, D1, K2, D2,
                        cv::Size(c.resolution.x(), c.resolution.y()), R, T, E, F,
                        CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_FIX_INTRINSIC |
                        CV_CALIB_FIX_K1 | CV_CALIB_FIX_K2,
                        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

    cv::cv2eigen(F, c.static_fundamental);
    Eigen::Matrix3f R_eig;
    Eigen::Vector3f T_eig;
    cv::cv2eigen(R, R_eig);
    cv::cv2eigen(T, T_eig);
    c.transform.matrix().topLeftCorner<3, 3>() = R_eig;
    c.transform.matrix().topRightCorner<3, 1>() = T_eig;
    std::cout << c.transform.matrix() << std::endl;
    exit(1);
}
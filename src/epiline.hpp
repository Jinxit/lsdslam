#pragma once

struct epiline
{
    epiline(const Eigen::Vector2i point, const Eigen::Vector3f line)
        : point(point), line(line) { };

    Eigen::Vector2i point;
    Eigen::Vector3f line;
};
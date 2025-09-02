#pragma once

#include <Eigen/Core>

struct Ray {
    Eigen::Vector3f origin;
    Eigen::Vector3f direction;
    Eigen::Vector3f throughput;

    Ray() {}
    Ray(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction)
        : origin(origin), direction(direction.normalized()) {}

    Eigen::Vector3f operator()(float t) const { return origin + t * direction; }

    // may eventually need some way to store and track mediums ...
};
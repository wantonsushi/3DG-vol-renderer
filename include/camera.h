#pragma once

#include <cmath>
#include <numbers>
#include "ray.h"

class Camera {
protected:
    Eigen::Vector3f position;
    Eigen::Vector3f view_dir;
    Eigen::Vector3f right;
    Eigen::Vector3f up;

public:
    Camera(const Eigen::Vector3f& position, const Eigen::Vector3f& view_dir)
        : position(position), view_dir(view_dir.normalized())
    {
        const Eigen::Vector3f world_up(0.0f, 1.0f, 0.0f);

        right = view_dir.cross(world_up).normalized();
        up = right.cross(view_dir).normalized();
    }

    virtual ~Camera() = default;

    virtual Ray sample_ray(const Eigen::Vector2d& uv) const = 0;
};

// ============================================================================================

class Pinhole_Camera : public Camera {
private:
    float fov;
    float focal_length;
    Eigen::Vector3f pinhole;

public:
    Pinhole_Camera(const Eigen::Vector3f& position, const Eigen::Vector3f& view_dir, float fov)
        : Camera(position, view_dir), fov(fov)
    {
        focal_length = 1.0f / std::tan(0.5f * fov);
        pinhole = position + focal_length * view_dir;
    }

    Ray sample_ray(const Eigen::Vector2d& uv) const override {
        // map uv from [0, 1] to [-1, 1] (+flipped)
        float u = 1.0f - static_cast<float>(uv.x()) * 2.0f;
        float v = static_cast<float>(uv.y()) * 2.0f - 1.0f;

        Eigen::Vector3f ray_origin = position + u * right + v * up;
        Eigen::Vector3f ray_dir = (pinhole - ray_origin);
        return Ray(ray_origin, ray_dir.normalized());
    }
};

// ============================================================================================

class Orthographic_Camera : public Camera {
public:
    Orthographic_Camera(const Eigen::Vector3f& position,
                       const Eigen::Vector3f& forward)
        : Camera(position, forward) {}

    Ray sample_ray(const Eigen::Vector2d& uv) const override {
        // map uv from [0, 1] to [-1, 1]
        float u = static_cast<float>(uv.x()) * 2.0f - 1.0f;
        float v = 1.0f - static_cast<float>(uv.y()) * 2.0f;

        Eigen::Vector3f ray_origin = position + u * right + v * up;
        Eigen::Vector3f ray_dir = view_dir;

        return Ray(ray_origin, ray_dir.normalized());
    }
};

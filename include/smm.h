#pragma once

#include "ray.h"
#include <numbers>
#include <Eigen/Geometry>

struct PrimitiveHitEvent {
    float t;        // distance along ray
    bool entering;  // entering/exiting primitive
    size_t index;   // index in mixture model array

    bool operator<(const PrimitiveHitEvent& other) const {
        return t < other.t;
    }
};

struct Sphere {
    const Eigen::Vector3f center;
    const float radius;
    const float sigma_a;
    const float sigma_s;
    //Eigen::Vector3f emission;       // Emission

    Sphere(const Eigen::Vector3f& center, float radius,
           float sigma_a = 0.0f, float sigma_s = 1.0f)
        : center(center), radius(radius), sigma_a(sigma_a), sigma_s(sigma_s) {}

    // compute ray-sphere intersection
    bool intersect(const Ray& ray, float& t_enter, float& t_exit) const {
        Eigen::Vector3f L = center - ray.origin;
        float tca = L.dot(ray.direction);
        float d2 = L.squaredNorm() - tca * tca;
        float r2 = radius * radius;
        if (d2 > r2) return false;
        float thc = std::sqrt(r2 - d2);
        t_enter = tca - thc;
        t_exit = tca + thc;
        return t_exit >= 0.0f; // at least one point is ahead of ray
    }
};

class SphereMixtureModel {
public:
    std::vector<Sphere> spheres;

    SphereMixtureModel() = default;

    explicit SphereMixtureModel(const std::vector<Sphere>& spheres)
        : spheres(spheres) {}

    size_t get_num_spheres() const { return spheres.size(); }

    // get all individual intersection events, sorted in ascending order
    std::vector<PrimitiveHitEvent> intersect_events(const Ray& ray) const {
        std::vector<PrimitiveHitEvent> events;
        for (size_t i = 0; i < spheres.size(); ++i) {
            float t_enter, t_exit;
            if (spheres[i].intersect(ray, t_enter, t_exit)) {
                if (t_enter >= 0.0f) events.push_back({t_enter, true, i});
                if (t_exit >= 0.0f) events.push_back({t_exit, false, i});
            }
        }
        std::sort(events.begin(), events.end());
        return events;
    }

     // evaluate absorption/scattering coefficients over multiple homogenous spheres    
    void evaluate_sigma(const std::vector<bool>& active, float& sigma_a, float& sigma_s) const {
        sigma_a = 0.0f;
        sigma_s = 0.0f;

        for (size_t i = 0; i < spheres.size(); ++i) {
            if (active[i]) {
                sigma_a += spheres[i].sigma_a;
                sigma_s += spheres[i].sigma_s;
            }
        }
    }

    // get transmittance over all active spheres
    float transmittance_from_events(const Ray& ray, const std::vector<PrimitiveHitEvent>& events, float tmax) const {
        float T = 1.0f;
        std::vector<bool> active(spheres.size(), false);

        float t_prev = 0.0f;

        for (const auto& e : events) {
            if (e.t > tmax) break;

            float dt = e.t - t_prev;
            float sigma_t = 0.0f;
            for (size_t i = 0; i < spheres.size(); ++i) {
                if (active[i]) {
                    sigma_t += spheres[i].sigma_a + spheres[i].sigma_s;
                }
            }

            T *= std::exp(-sigma_t * dt);

            active[e.index] = e.entering;
            t_prev = e.t;
        }

        return T;
    }
};


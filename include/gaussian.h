#pragma once

#include "ray.h"
#include <numbers>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <iostream>

class Gaussian {
private:
    Eigen::Vector3f mean;           // Centre of the Gaussian
    Eigen::Matrix3f covariance;     // Covariance matrix defining shape
    float density;                  // Density scale
    float albedo;                   // Single scattering albedo
    Eigen::Vector3f emission;       // Emitted radiance (default ⟨0,0,0⟩)

    const float R = 3.0f;                 // Threshold for finite bound (R*STDDEV)

    // precomputable quantities:
    Eigen::Matrix3f inv_cov;         
    float norm;
    
    Eigen::Matrix3f whitening_T;
    Eigen::Matrix3f whitening_T_inv;

public:
    Gaussian(
        const Eigen::Vector3f& mean,
        const Eigen::Matrix3f& covariance,
        float density,
        float albedo,
        const Eigen::Vector3f& emission = Eigen::Vector3f::Zero())
        : mean(mean), covariance(covariance), density(density), albedo(albedo), emission(emission)
    {
        inv_cov = covariance.inverse();
        float det_cov = covariance.determinant();
        norm = std::pow(2.0f * std::numbers::pi, -1.5f) * std::pow(det_cov, -0.5f);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covariance);
        Eigen::Vector3f eigvals = es.eigenvalues();
        Eigen::Matrix3f eigvecs = es.eigenvectors();

        Eigen::Matrix3f sqrt_inv_lambda = Eigen::Matrix3f::Zero();
        sqrt_inv_lambda(0,0) = 1.0f / std::sqrt(eigvals[0]);
        sqrt_inv_lambda(1,1) = 1.0f / std::sqrt(eigvals[1]);
        sqrt_inv_lambda(2,2) = 1.0f / std::sqrt(eigvals[2]);

        whitening_T = sqrt_inv_lambda * eigvecs.transpose();
        whitening_T_inv = whitening_T.inverse();
    }

    float evaluate(const Eigen::Vector3f& x) const {
        Eigen::Vector3f d = x - mean;
        float exponent = -0.5f * d.transpose() * inv_cov * d;
        return norm * std::exp(exponent);
    }

    float mu_t(const Eigen::Vector3f& x) const { return density * evaluate(x); }  // get extinction at a point
    float get_albedo() const { return albedo; }                                   // get single scattering scattering albedo

    // analytic ray-gaussian intersection by solving ellipsoid equation
    bool intersect(const Ray& ray, float& t_enter, float& t_exit) const {
        // shift origin into Gaussian's local space
        Eigen::Vector3f p = ray.origin - mean;

        // solving Ellipsoid equation is a quadratic:
        // inverse covariance matrix precomputed for efficiency (see CTOR)

        // quadratic coefficients
        float A = ray.direction.transpose() * inv_cov * ray.direction;
        float B = 2.0f * p.transpose() * inv_cov * ray.direction;
        float C = p.transpose() * inv_cov * p - (R * R);

        // solve quadratic equation
        float discriminant = B * B - 4.0f * A * C;
        if (discriminant < 0.0f) {
            return false;  // no intersection
        }

        float sqrtD = std::sqrt(discriminant);

        // two solutions for t
        float t0 = (-B - sqrtD) / (2.0f * A);
        float t1 = (-B + sqrtD) / (2.0f * A);

        // ensure t0 <= t1
        if (t0 > t1) std::swap(t0, t1);

        // discard intersections completely behind ray
        if (t1 < 0.0f) return false;

        // clamp entry/exit to be in front of ray origin
        t_enter = (t0 >= 0.0f) ? t0 : 0.0f;
        t_exit  = t1;

        return true;
    }

    // compute optical depth over the gaussian
    float optical_depth(const Ray& ray, float t0, float t1) const {
        // shift to local coords
        Eigen::Vector3f p = ray.origin - mean;
        Eigen::Vector3f d  = ray.direction;

        // set up quadratic: a t^2 + b t + c
        float a = d.transpose() * inv_cov * d;
        float b = 2.0f * p.transpose() * inv_cov * d;
        float c = p.transpose() * inv_cov * p;

        // prefactor:
        float pref = density * norm * std::sqrt(std::numbers::pi / (2.0f * a));

        auto F = [&](float t) {
            // argument of erf:
            float arg = (b + 2.0f * a * t) / (2.0f * std::sqrt(2.0f * a));
            return std::erf(arg);
        };

        // compute difference of erf
        return pref * std::exp(-0.5f * (c - b*b/(4.0f*a))) * (F(t1) - F(t0));
    }

    // alternative ray-gaussian intersection by using a whitening transform
    // NOT WORKING, slower anyways...
    bool intersect_sphere_map(const Ray& ray, float& t_enter, float& t_exit) const {
        // shift origin into Gaussian's local space
        Eigen::Vector3f p_local = ray.origin - mean;
        Eigen::Vector3f d_local = ray.direction;

        // get whitening transform to sphere space
        // tranformation precomputed for efficiency (see CTOR)

        Eigen::Vector3f p_sphere = whitening_T * p_local;
        Eigen::Vector3f d_sphere = whitening_T * d_local;

        // standard sphere intersection
        Eigen::Vector3f L = -p_sphere;
        float tca = L.dot(d_sphere);
        float d2 = L.squaredNorm() - tca * tca;
        float r2 = R * R;
        if (d2 > r2) return false;
        float thc = std::sqrt(r2 - d2);
        float t0_s = tca - thc;
        float t1_s = tca + thc;
        if (t1_s < 0.0f) return false;

        if (t0_s > t1_s) std::swap(t0_s, t1_s);

        // intersection points in sphere space
        Eigen::Vector3f p0_s = p_sphere + t0_s * d_sphere;
        Eigen::Vector3f p1_s = p_sphere + t1_s * d_sphere;

        // map intersection points back to Gaussian/world space
        Eigen::Vector3f p0 = mean + whitening_T_inv * p0_s;
        Eigen::Vector3f p1 = mean + whitening_T_inv * p1_s;

        // compute t along original ray: project (p - ray.origin) onto ray.direction
        float t0 = (p0 - ray.origin).dot(ray.direction);
        float t1 = (p1 - ray.origin).dot(ray.direction);

        if (t1 < 0.0f) return false; // both behind

        if (t0 > t1) std::swap(t0, t1);

        t_enter = (t0 >= 0.0f) ? t0 : 0.0f;
        t_exit = t1;

        return true;
    }
};
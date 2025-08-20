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

        whitening_T = sqrt_inv_lambda * eigvecs.transpose() * (1.0f / R);
    }

    float evaluate(const Eigen::Vector3f& x) const {
        Eigen::Vector3f d = x - mean;
        float exponent = -0.5f * d.transpose() * inv_cov * d;
        return norm * std::exp(exponent);
    }

    float mu_t(const Eigen::Vector3f& x) const { return density * evaluate(x); }  // get extinction at a point
    float get_albedo() const { return albedo; }                                   // get single scattering scattering albedo

    // analytic ray-gaussian intersection by solving ellipsoid equation
    bool intersect_direct(const Ray& ray, float& t_enter, float& t_exit) const {
        // shift origin into Gaussian's local space
        Eigen::Vector3f p = ray.origin - mean;

        // solving Ellipsoid equation is a quadratic:
        // inverse covariance matrix precomputed for efficiency (see CTOR)

        Eigen::Vector3f Md = inv_cov * ray.direction; 
        Eigen::Vector3f Mp = inv_cov * p;        

        // quadratic coefficients
        float A = ray.direction.dot(Md);
        float B = 2.0f * p.dot(Md);       
        float C = p.dot(Mp) - (R * R); 

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

        // clamp enter to zero if behind origin
        t_enter = (t0 >= 0.0f) ? t0 : 0.0f;
        t_exit  = t1;

        return true;
    }

    // compute optical depth over the gaussian
    float optical_depth(const Ray& ray, float t0, float t1) const {
        // shift origin into Gaussian's local space
        Eigen::Vector3f p = ray.origin - mean;

        // set up quadratic: a t^2 + b t + c
        float A = ray.direction.transpose() * inv_cov * ray.direction;
        float B = 2.0f * p.transpose() * inv_cov * ray.direction;
        float C = p.transpose() * inv_cov * p;

        // prefactor:
        float pref = density * norm * std::sqrt(std::numbers::pi / (2.0f * A));

        auto F = [&](float t) {
            // argument of erf:
            float arg = (B + 2.0f * A * t) / (2.0f * std::sqrt(2.0f * A));
            return std::erf(arg);
        };

        // compute difference of erf
        return pref * std::exp(-0.5f * (C - B*B/(4.0f*A))) * (F(t1) - F(t0));
    }

    // alternative ray-gaussian intersection by using a whitening transform
    bool intersect_whitening(const Ray& ray, float& t_enter, float& t_exit) const {
        // transform ray into whitened space
        // whitening transform maps R*stddev ellipsoid -> unit sphere
        // precomputed for efficiency (see CTOR)

        Eigen::Vector3f o_local = ray.origin - mean;
        Eigen::Vector3f o_w = whitening_T * o_local;
        Eigen::Vector3f d_w = whitening_T * ray.direction;

        // quadratic coefficients
        float A = d_w.dot(d_w);
        float B = 2.0f * o_w.dot(d_w);
        float C = o_w.dot(o_w) - 1.0f;

        // solve quadratic equation
        float discriminant = B * B - 4.0f * A * C;
        if (discriminant < 0.0f) {
            return false; // no intersection
        }

        float sqrtD = std::sqrt(discriminant);

        // two solutions for t
        float t0 = (-B - sqrtD) / (2.0f * A);
        float t1 = (-B + sqrtD) / (2.0f * A);

        // ensure t0 <= t1
        if (t0 > t1) std::swap(t0, t1);

        // discard intersections completely behind ray
        if (t1 < 0.0f) return false;

        // clamp enter to zero if behind origin
        t_enter = (t0 >= 0.0f) ? t0 : 0.0f;
        t_exit  = t1;

        return true;
    }
};
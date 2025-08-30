#pragma once

#include "scene.h"
#include "ray.h"

#include <vector>
#include <random>

#include <atomic>
#include <cstdint>

#include "rng.h"

inline bool has_nan(const Eigen::Vector3f& v) {
    return std::isnan(v.x()) || std::isnan(v.y()) || std::isnan(v.z());
}

// ============================================================================================
//  DISTANCE SOLVERS:
//      find t in [ta, tb] so that
//      accumulated optical depth from ta to t equals target
// ============================================================================================

// bisection solver
inline float solve_distance_bisection(
    const Ray &ray,
    float ta, float tb,
    const std::vector<size_t>& active_idxs,
    float target,
    const GaussianMixtureModel &gmm,
    int max_iters = 15,
    float tol = 1e-6f           // tolerance on optical depth residual
) {
    float a = ta, b = tb;

    for (int i = 0; i < max_iters; ++i) {
        float m = 0.5f * (a + b);
        float tau = 0.0f;
        for (auto idx : active_idxs)
            tau += gmm.gaussians[idx].optical_depth(ray, ta, m);
        
        float f = tau - target;
        if (std::fabs(f) <= tol) {
            //std::cerr << "stopped early on iter: " << i << std::endl;
            return m;
        }

        if (f < 0.0f) {
            a = m;
        } else {
            b = m;
        }
    }

    // give midpoint if max_iters reached
    return 0.5f * (a + b);
}

// ============================================================================================

// Newton-Raphson solver with fallback to bisection
inline float solve_distance_newton_raphson(
    const Ray &ray,
    float ta, float tb,
    const std::vector<size_t>& active_idxs,
    float target_tau,
    const GaussianMixtureModel &gmm,
    int max_iters = 8,
    float tol = 1e-6f
) {
    // initial guess: midpoint (this is what DSYG suggests)
    float a = ta, b = tb;
    float t = 0.5f * (a + b);

    auto compute_f = [&](float tt) -> float {
        // f(tt) = sum_i tau_i(ta, min(tt, tb)) - target_tau
        float sum_tau = 0.0f;
        float tt_clamped = std::min(tt, b);
        for (auto idx : active_idxs)
            sum_tau += gmm.gaussians[idx].optical_depth(ray, ta, tt_clamped);
        return sum_tau - target_tau;
    };

    for (int iter = 0; iter < max_iters; ++iter) {
        float f = compute_f(t);
        if (std::fabs(f) <= tol) {
            //std::cerr << "converged on iter: " << iter << std::endl;
            return std::clamp(t, a, b);
        }

        // numeric derivative via forward difference
        float h = std::max(1e-5f, (b - a) * 1e-6f);
        float tp = std::min(b, t + h);
        float fp = compute_f(tp);

        float deriv = (fp - f) / (tp - t);

        // fallback if derivative non-positive, zero, NaN, or too small (doesn't seem to happen)
        if (!(deriv > 0.0f) || !std::isfinite(deriv) || std::fabs(deriv) < 1e-12f) {
            //std::cerr << "derivative non-positive, zero, NaN, or too small" << std::endl;
            return solve_distance_bisection(ray, ta, tb, active_idxs, target_tau, gmm);
        }

        // Newton step
        float t_next = t - f / deriv;

        // fallback if stepped out of segment bounds
        if (!std::isfinite(t_next) || t_next < a || t_next > b) {
            //std::cerr << "stepped out of bounds" << std::endl;
            //t_next = std::clamp(t_next, a, b);
            return solve_distance_bisection(ray, ta, tb, active_idxs, target_tau, gmm);
        }

        // small change -> converged
        if (std::fabs(t_next - t) <= tol * std::max(1.0f, std::fabs(t))) {
            t = t_next;
            //std::cerr << "converged on iter: " << iter << std::endl;
            return std::clamp(t, a, b);
        }

        t = t_next;
    }

    // If we exit loop with no convergence -> fallback to bisection
    //std::cerr << "did not converge" << std::endl;
    return solve_distance_bisection(ray, ta, tb, active_idxs, target_tau, gmm);
}

// ============================================================================================

// Uniform approximation over critical segment
inline float solve_distance_uniform(
    float ta, float tb
) {
    float u = rand01();
    return ta + u * (tb - ta);
}


// ========================================================================================================

// Exactly **one** of these must be defined
//#define BISECTION
//#define NEWTON
//#define ANALYTIC_PLUS_BISECTION
#define ANALYTIC_PLUS_NEWTON
//#define UNIFORM

// Abstract distance solver:
inline float solve_distance(
    const Ray &ray,
    float ta, float tb,
    const std::vector<size_t>& active_idxs,
    float remaining_tau,
    const GaussianMixtureModel &gmm
) {
#if defined(UNIFORM)
    return solve_distance_uniform(ta, tb);

#elif defined(BISECTION)
    return solve_distance_bisection(ray, ta, tb, active_idxs, remaining_tau, gmm);

#elif defined(NEWTON)
    return solve_distance_newton_raphson(ray, ta, tb, active_idxs, remaining_tau, gmm);

#elif defined(ANALYTIC_PLUS_BISECTION)
    if (active_idxs.size() == 1) {
        const auto &G = gmm.gaussians[active_idxs[0]];
        float t_analytic = 0.0f;
        if (G.solve_for_t_given_tau(ray, ta, tb, remaining_tau, t_analytic)) {
            return std::clamp(t_analytic, ta, tb);
        }
    }
    return solve_distance_bisection(ray, ta, tb, active_idxs, remaining_tau, gmm);

#elif defined(ANALYTIC_PLUS_NEWTON)
    if (active_idxs.size() == 1) {
        const auto &G = gmm.gaussians[active_idxs[0]];
        float t_analytic = 0.0f;
        if (G.solve_for_t_given_tau(ray, ta, tb, remaining_tau, t_analytic)) {
            return std::clamp(t_analytic, ta, tb);
        }
        std::cerr << "analytic failed" << std::endl;
    }
    return solve_distance_newton_raphson(ray, ta, tb, active_idxs, remaining_tau, gmm);
#endif
}

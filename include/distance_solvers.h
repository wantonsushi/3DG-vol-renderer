#pragma once

#include "scene.h"
#include "ray.h"

#include <vector>

// ============================================================================================
//  DISTANCE SOLVERS:
//      find t in [ta, tb] so that
//      accumulated optical depth from ta to t equals target
// ============================================================================================

// bisection solver
float solve_distance_bisection(
    const Ray &ray,
    float ta, float tb,
    const std::vector<size_t>& active_idxs,
    float target,
    const GaussianMixtureModel &gmm,
    int max_iters = 15,
    float tol_tau = 1e-6f           // tolerance on optical depth residual
) {
    float a = ta, b = tb;

    for (int i = 0; i < max_iters; ++i) {
        float m = 0.5f * (a + b);
        float tau = 0.0f;
        for (auto idx : active_idxs)
            tau += gmm.gaussians[idx].optical_depth(ray, ta, m);
        
        float f = tau - target;
        if (std::fabs(f) <= tol_tau) {
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

// Newton-Raphson solver with fallback to bisection
float solve_distance_newton_raphson(
    const Ray &ray,
    float ta, float tb,
    const std::vector<size_t>& active_idxs,
    float target_tau,
    const GaussianMixtureModel &gmm,
    int max_iters = 8,
    float tol = 1e-6f
) {
    // initial guess: midpoint (paper suggests half the segment length)
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

        // numeric derivative via central difference (robust when you don't have direct kernel eval)
        float h = std::max(1e-5f, (b - a) * 1e-6f); // relative small step
        float tp = std::min(b, t + h);
        float tm = std::max(a, t - h);
        float fp = compute_f(tp);
        float fm = compute_f(tm);

        float deriv = (fp - fm) / (tp - tm);

        // if derivative non-positive, zero, NaN, or too small -> fail and fallback
        if (!(deriv > 0.0f) || !std::isfinite(deriv) || std::fabs(deriv) < 1e-12f) {
            // fallback
            //std::cerr << "derivative non-positive, zero, NaN, or too small" << std::endl;
            return solve_distance_bisection(ray, ta, tb, active_idxs, target_tau, gmm, 15);
        }

        // Newton step
        float t_next = t - f / deriv;

        // Clip to segment bounds (paper suggests clipping)
        if (!std::isfinite(t_next) || t_next < a || t_next > b) {
            // If it steps out-of-bounds or becomes invalid -> fallback
            //std::cerr << "stepped out of bounds" << std::endl;
            return solve_distance_bisection(ray, ta, tb, active_idxs, target_tau, gmm, 15);
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
    return solve_distance_bisection(ray, ta, tb, active_idxs, target_tau, gmm, 15);
}

// Linear-homogeneous approximation over a segment using accumulated tau
// target_tau = accum_tau + linear interpolation over segment [ta, tb] with total seg_tau
float solve_distance_homogeneous(
    float ta, float tb,
    float accum_tau,           // optical depth accumulated before ta
    float seg_tau,             // total optical depth of segment [ta, tb]
    float target_tau          // total optical depth to reach
) {
    // linear interpolation
    float t = ta + (tb - ta) * (target_tau - accum_tau) / seg_tau;
    return std::clamp(t, ta, tb);
}

// ========================================================================================================

// Exactly **one** of these must be defined
// #define BISECTION
// #define NEWTON
// #define ANALYTIC_PLUS_BISECTION
 #define ANALYTIC_PLUS_NEWTON
// define HOMOGENEOUS

// Abstract distance solver:
float solve_distance(
    const Ray &ray,
    float ta, float tb,
    const std::vector<size_t>& active_idxs,
    float accum_tau,
    float seg_tau,
    float target_tau,
    const GaussianMixtureModel &gmm
) {
    float remaining_tau = target_tau - accum_tau;

#if defined(HOMOGENEOUS)
    return solve_distance_homogeneous(ta, tb, accum_tau, seg_tau, target_tau);

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

#else
    #error "Must define a distance solver"
#endif
}

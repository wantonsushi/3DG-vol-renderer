#pragma once

#include <vector>
#include "gaussian.h"
#include "smm.h"

class GaussianMixtureModel {
public:
    std::vector<Gaussian> gaussians;

    GaussianMixtureModel() = default;

    explicit GaussianMixtureModel(const std::vector<Gaussian>& gs)
        : gaussians(gs) {}

    size_t get_num_gaussians() const {
        return gaussians.size();
    }

    // get all individual intersection events, sorted in ascending order
    std::vector<PrimitiveHitEvent> intersect_events(const Ray& ray) const {
        std::vector<PrimitiveHitEvent> events;
        for (size_t i = 0; i < gaussians.size(); ++i) {
            float t0, t1;
            //if (gaussians[i].intersect_whitening(ray, t0, t1)) {
            if (gaussians[i].intersect_direct(ray, t0, t1)) {
                if (t0 >= 0.0f) events.push_back({ t0, true,  i });
                if (t1 >= 0.0f) events.push_back({ t1, false, i });
            }
        }
        std::sort(events.begin(), events.end());
        return events;
    }

    // evaluate absorption/scattering coefficients over multiple spatially varying gaussians    
    void evaluate_sigma(
        const std::vector<bool>& active,
        const Eigen::Vector3f&   pos,
        float&                   sigma_a,
        float&                   sigma_s
    ) const {
        float sum_mu_t       = 0.0f;
        float sum_mu_t_alb   = 0.0f;

        // 1) accumulate extinction and albedo
        for (size_t i = 0; i < gaussians.size(); ++i) {
            if (!active[i]) continue;
            float mu_t_i = gaussians[i].mu_t(pos);
            sum_mu_t     += mu_t_i;
            sum_mu_t_alb += mu_t_i * gaussians[i].get_albedo();
        }

        if (sum_mu_t <= 0.0f) {
            sigma_s = sigma_a = 0.0f;
            return;
        }

        // 2) absorption coefficient
        float a_mix = sum_mu_t_alb / sum_mu_t;

        // 3) scattering coefficient
        sigma_s = a_mix       * sum_mu_t;
        sigma_a = (1.0f - a_mix) * sum_mu_t;
    }

    // get transmittance over all active gaussians = exp(-sum(optical depths))
    float transmittance_over_segment(
        const Ray& ray,
        float t0,
        float t1,
        const std::vector<size_t>& active_indices
    ) const {
        float optical_depth_sum = 0.0f;
        for (size_t i : active_indices) {
            optical_depth_sum += gaussians[i].optical_depth(ray, t0, t1);
        }
        return std::exp(-optical_depth_sum);
    }

    // much faster transmittance along ray from t=0 to t = tmax
    // no sorting, for single-scatter shadow transmittance/next-event-estimation
    float transmittance_up_to(const Ray &ray, float tmax) const {
        if (tmax <= 0.0f) return 1.0f;

        double optical_depth_sum = 0.0; // use double for accumulate safety
        for (size_t i = 0; i < gaussians.size(); ++i) {
            float g_t0, g_t1;
            // only consider this gaussian if it intersects the ray
            if (!gaussians[i].intersect_direct(ray, g_t0, g_t1))
                continue;

            // clip to [0, tmax]
            float a = std::max(0.0f, g_t0);
            float b = std::min(tmax, g_t1);
            if (b > a) {
                optical_depth_sum += gaussians[i].optical_depth(ray, a, b);
            }
        }

        return std::exp(-float(optical_depth_sum));
    }
};
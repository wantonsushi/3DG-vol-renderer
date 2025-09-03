#pragma once

#include "gmm.h"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// Standard ADAM
// ==========================================================================

class AdamOptimizer {
public:
    AdamOptimizer(
        size_t ndim,
        float lr = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f
    )
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0)
    {
        m.assign(ndim, 0.0f);
        v.assign(ndim, 0.0f);
    }

    // step: modifies params in-place using grads (same size as params)
    // returns false if sizes mismatch
    bool step(std::vector<float>& params, const std::vector<float>& grads) {
        if (params.size() != grads.size() || params.size() != m.size() || params.size() != v.size()) return false;
        ++t;
        const float a = lr * std::sqrt(1.0f - std::pow(beta2, (float)t)) / (1.0f - std::pow(beta1, (float)t));
        for (size_t i = 0; i < params.size(); ++i) {
            float g = grads[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            params[i] -= a * (m[i] / (std::sqrt(v[i]) + eps));
        }
        return true;
    }

    void reset_state() {
        std::fill(m.begin(), m.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        t = 0;
    }

    size_t dim() const { return m.size(); }

private:
    std::vector<float> m, v;        // moments
    float lr, beta1, beta2, eps;
    int t;
};



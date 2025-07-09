#pragma once
#include <vector>
#include "gaussian.h"


struct Volume {
    // Computes μ_s, μ_a at a point or segment from active Gaussians
    void eval_coefficients(const std::vector<const Gaussian*>& active,
                           float t, float& mu_s, float& mu_a) const;
};
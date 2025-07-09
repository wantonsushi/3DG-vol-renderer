#pragma once
#include "ray.h"


struct Gaussian {
    // Mean, covariance, scattering/absorption coefficients
    bool intersect(const Ray& ray, float& t_enter, float& t_exit) const;
};
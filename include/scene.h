#pragma once
#include <vector>
#include "gaussian.h"


struct Scene {
    std::vector<Gaussian> gaussians;
    static Scene load_from_file(const std::string& path);
};
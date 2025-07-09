#pragma once
#include "scene.h"


class Renderer {
public:
    Renderer(int w, int h);
    void render(const Scene& scene, std::vector<Pixel>& framebuffer);
};
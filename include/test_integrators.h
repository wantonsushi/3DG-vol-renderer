#pragma once

#include "integrator.h"

// integrators not included in report

// ============================================================================================
//  RAY-MARCHING FOR SPHERES ONLY (ANALYTICAL TRANSMITTANCE) (SINGLE SCATTERING)
// ============================================================================================

class RayMarchingSpheres : public Integrator {
private:
    float step_size;
    int env_samples;

public:
    RayMarchingSpheres(const std::shared_ptr<Camera>& camera, float step_size = 0.01f, int env_samples = 5)
        : Integrator(camera), step_size(step_size), env_samples(env_samples) 
        {
            std::cout << "Sphere Raymarcher (with analytical transmittance)" << std::endl;
        }

    void render(const Scene& scene, Image& image) override {
        const unsigned int width = image.get_width();
        const unsigned int height = image.get_height();

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {

                float u = (i + 0.5f) / width;
                float v = (j + 0.5f) / height;

                Eigen::Vector2d uv(u, v);
                Ray ray = camera->sample_ray(uv);

                Eigen::Vector3f radiance = Eigen::Vector3f::Zero();

                const auto events = scene.intersect_events(ray);
                if (events.empty()) {
                    image.set_pixel(i, j, scene.env_color);
                    continue;
                }

                // setup marching loop
                std::vector<bool> active(scene.get_num_primitives(), false);

                size_t current_event = 0;
                float t = 0.0f;
                float T = 1.0f;

                float t_end = events.back().t;

                while (t < t_end) {
                    // manage active spheres
                    while (current_event < events.size() && events[current_event].t <= t) {
                        active[events[current_event].index] = events[current_event].entering;
                        current_event++;
                    }

                    Eigen::Vector3f pos = ray.origin + t * ray.direction;

                    float sigma_a, sigma_s;
                    scene.evaluate_sigma(active, pos, sigma_a, sigma_s);

                    if (sigma_s > 0.0f) {
                        // get Li using point lights
                        Eigen::Vector3f Li = Eigen::Vector3f::Zero();

                        for (const auto& light : scene.lights) {
                            Eigen::Vector3f wi = (light.position - pos).normalized();
                            float dist = (light.position - pos).norm();

                            Ray shadow_ray(pos, wi);
                            auto shadow_events = scene.intersect_events(shadow_ray);

                            for (size_t i = 0; i < active.size(); ++i) {
                                if (active[i]) {
                                    // t = 0, entering = true, index = i
                                    shadow_events.insert(shadow_events.begin(), { 0.0f, true, i });
                                }
                            }

                            // get analytic transmittance
                            float Tr = scene.transmittance_from_events(shadow_ray, shadow_events, dist);

                            Li += Tr * light.intensity / (dist * dist);
                        }

                        // --- add Monte Carlo‐sampled environment ---
                        Eigen::Vector3f Le_env = Eigen::Vector3f::Zero();
                        for (int s = 0; s < env_samples; ++s) {
                            Eigen::Vector3f wi = sample_uniform_direction_old();
                            Ray shadow(pos, wi);

                            // get analytic transmittance
                            auto shadow_ev = scene.intersect_events(shadow);

                            for (size_t i = 0; i < active.size(); ++i) {
                                if (active[i]) {
                                    // t = 0, entering = true, index = i
                                    shadow_ev.insert(shadow_ev.begin(), { 0.0f, true, i });
                                }
                            }

                            float Tr_env = scene.transmittance_from_events(shadow, shadow_ev, std::numeric_limits<float>::infinity());

                            //std::cout << Tr_env << std::endl;
                            Le_env += Tr_env * scene.env_color;

                        }
                        Le_env /= float(env_samples);
                        Le_env *= (4.0f * std::numbers::pi);

                        //std::cout << "Li: " << Li << std::endl;
                        //std::cout << "Le: " << Le_env << std::endl;

                        Eigen::Vector3f contrib = T * sigma_s * (Li + Le_env) * step_size * (1.0f / (4.0f * std::numbers::pi));
                        radiance += contrib;
                    }

                    T *= std::exp(-step_size * (sigma_a + sigma_s));
                    t += step_size;
                }

                radiance += T * scene.env_color;

                image.set_pixel(i, j, radiance);
            }
        }
    }
};


// ============================================================================================
//  RAY-MARCHING GAUSSIANS (ANALYTIC TRANSMITTANCE) (SINGLE SCATTERING)
// ============================================================================================

class RayMarchingGaussians : public Integrator {
private:
    float step_size;
    int   env_samples;

public:
    RayMarchingGaussians(
        const std::shared_ptr<Camera>& camera,
        float step_size   = 0.01f,
        int   env_samples = 20
    ) : Integrator(camera),
        step_size(step_size),
        env_samples(env_samples)
    {
        std::cout << "Gaussian Raymarcher (with analytical transmittance)" << std::endl;
    }

    void render(const Scene& scene, Image& image) override {
        const int W = (int)image.get_width();
        const int H = (int)image.get_height();

        #pragma omp parallel for collapse(2) schedule(dynamic,1)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {

                float u = (x + 0.5f)/W, v = (y + 0.5f)/H;
                Ray ray = camera->sample_ray({u,v});
                auto events = scene.intersect_events(ray);
                if (events.empty()) {
                    image.set_pixel(x,y, scene.env_color);
                    continue;
                }

                float t_end = events.back().t;

                // setup marching loop
                std::vector<bool> active(scene.get_num_primitives(), false);
                size_t evt_i = 0;

                float t = 0.0f;
                float T = 1.0f;
                Eigen::Vector3f L = Eigen::Vector3f::Zero();

                while (t < t_end) {
                    // manage active gaussians
                    while (evt_i < events.size() && events[evt_i].t <= t) {
                        active[ events[evt_i].index ] = events[evt_i].entering;
                        ++evt_i;
                    }

                    Eigen::Vector3f pos = ray.origin + t * ray.direction;
                    float sigma_a, sigma_s;
                    scene.evaluate_sigma(active, pos, sigma_a, sigma_s);

                    if (sigma_s > 0.0f) {
                        // --- point lights ---
                        Eigen::Vector3f Li = Eigen::Vector3f::Zero();
                        for (auto const& light : scene.lights) {
                            Eigen::Vector3f wi   = (light.position - pos).normalized();
                            float           dist = (light.position - pos).norm();
                            Ray   shadow_ray(pos, wi);

                            auto shadow_ev = scene.intersect_events(shadow_ray);
                            for (size_t i = 0; i < active.size(); ++i)
                                if (active[i])
                                    shadow_ev.insert(shadow_ev.begin(), {0.0f, true, i});

                            // ... I should have abstracted this...
                            // get analytic transmittance to the light
                            std::vector<size_t> active_idxs;
                            std::vector<bool> mask = active;
                            float t_prev = 0;
                            size_t se_i = 0;
                            float Tr = 1.0f;
                            while (t_prev < dist) {
                                float t_next = (se_i < shadow_ev.size() ? shadow_ev[se_i].t : dist);
                                // gather active indices
                                active_idxs.clear();
                                for (size_t i = 0; i < mask.size(); ++i)
                                    if (mask[i]) active_idxs.push_back(i);
                                // analytic trans over [t_prev, t_next]
                                Tr *= scene.gmm->at(0).transmittance_over_segment(
                                          shadow_ray, t_prev, t_next, active_idxs);
                                // advance and flip that event
                                if (se_i < shadow_ev.size()) {
                                    mask[ shadow_ev[se_i].index ] = shadow_ev[se_i].entering;
                                    ++se_i;
                                }
                                t_prev = t_next;
                            }

                            Li += Tr * light.intensity / (dist*dist);
                        }

                        // --- environment ---
                        Eigen::Vector3f Le = Eigen::Vector3f::Zero();
                        for (int si = 0; si < env_samples; ++si) {
                            Eigen::Vector3f wi = sample_uniform_direction_old();
                            Ray               env_ray(pos, wi);

                            auto env_ev = scene.intersect_events(env_ray);
                            for (size_t i = 0; i < active.size(); ++i)
                                if (active[i])
                                    env_ev.insert(env_ev.begin(), {0.0f, true, i});

                            // get analytic transmittance to environment
                            std::vector<size_t> active_idxs;
                            std::vector<bool> mask = active;
                            float t_prev = 0,
                                  Tr_env = 1.0f;
                            size_t ei = 0;
                            while (ei < env_ev.size()) {
                                float t_next = env_ev[ei].t;
                                // compute active
                                active_idxs.clear();
                                for (size_t i = 0; i < mask.size(); ++i)
                                    if (mask[i]) active_idxs.push_back(i);
                                // get analytic transmittance
                                Tr_env *= scene.gmm->at(0)
                                           .transmittance_over_segment(env_ray, t_prev, t_next, active_idxs);
                                // advance and flip
                                mask[ env_ev[ei].index ] = env_ev[ei].entering;
                                t_prev = t_next;
                                ++ei;
                            }
                            Le += Tr_env * scene.env_color;
                        }
                        Le = (Le / float(env_samples)) * (4.0f * std::numbers::pi);

                        L += T * sigma_s * (Li + Le) 
                             * step_size * (1.0f / (4.0f * std::numbers::pi));
                    }

                    // update T over raymarching step segment analytically:
                    std::vector<size_t> active_idxs;
                    for (size_t i = 0; i < active.size(); ++i)
                        if (active[i]) active_idxs.push_back(i);
                    // get segment transmittance
                    float segmentTr = scene.gmm->at(0)
                        .transmittance_over_segment(ray, t, t+step_size, active_idxs);
                    T *= segmentTr;

                    t += step_size;
                }

                L += T * scene.env_color;
                image.set_pixel(x,y, L);
            }
        }
    }
};

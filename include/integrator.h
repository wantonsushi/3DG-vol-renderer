#pragma once

#include <omp.h>
#include <memory>

#include "scene.h"
#include "image.h"
#include "camera.h"
#include "distance_solvers.h"

#include <iostream>

inline Eigen::Vector3f sample_uniform_direction_old() {
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float xi1 = dist(gen);
    float xi2 = dist(gen);

    float theta = 2.0f * std::numbers::pi * xi1;        // azimuth
    float phi = std::acos(1.0f - 2.0f * xi2);           // polar

    float x = std::sin(phi) * std::cos(theta);
    float y = std::sin(phi) * std::sin(theta);
    float z = std::cos(phi);

    return Eigen::Vector3f(x, y, z);
}


// generate a uniform random direction over the unit sphere
inline Eigen::Vector3f sample_uniform_direction(PCG32& rng) {
    float xi1 = rng.uniform();
    float xi2 = rng.uniform();

    float theta = 2.0f * std::numbers::pi * xi1;        // azimuth
    float phi = std::acos(1.0f - 2.0f * xi2);           // polar

    float x = std::sin(phi) * std::cos(theta);
    float y = std::sin(phi) * std::sin(theta);
    float z = std::cos(phi);

    return Eigen::Vector3f(x, y, z);
}

// ============================================================================================

// abstract class
class Integrator {
protected:
    const std::shared_ptr<Camera> camera;

public:
    Integrator(const std::shared_ptr<Camera> &camera) : camera(camera) {}

    virtual void render(const Scene& scene, Image &image) = 0;
};

// spagetti code incoming ...

// ============================================================================================
//  TEST SPHERE/GAUSSIAN INTERSECTION
// ============================================================================================

class TestIntegrator : public Integrator {
public:
    TestIntegrator(const std::shared_ptr<Camera>& camera)
        : Integrator(camera) {}

    void render(const Scene& scene, Image& image) override {
        const unsigned int width = image.get_width();
        const unsigned int height = image.get_height();

    //#pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                float u = (i + 0.5f) / width;
                float v = (j + 0.5f) / height;

                Eigen::Vector2d uv(u, v);
                Ray ray = camera->sample_ray(uv);

                // set some constant color if an intersection is detected on the primary ray
                std::vector<PrimitiveHitEvent> hits;
                scene.intersect_events(ray, hits);
                Eigen::Vector3f color = !hits.empty()
                    ? Eigen::Vector3f(1.0f, 0.0f, 1.0f)
                    : scene.env_color;                   

                image.set_pixel(i, j, color);
            }
        }
    }
};

// ============================================================================================
//  RAY-MARCHING ANY MEDIUM (RAY-MARCHING TRANSMITTANCE) (SINGLE SCATTERING)
// ============================================================================================

class PureRayMarching : public Integrator {
private:
    float step_size;
    int env_samples;

    float march_transmittance(
        const Scene& scene,
        const Ray& ray,
        float t_max,
        const std::vector<PrimitiveHitEvent>& events,
        const std::vector<bool>& active0,
        float step_size
    ) {
        std::vector<bool> active = active0;
        float T = 1.0f;
        size_t evt_idx = 0, n_evt = events.size();

        // march in [0, t_max)
        for (float t = 0.0f; t < t_max; t += step_size) {
            // manage active primitives
            while (evt_idx < n_evt && events[evt_idx].t <= t) {
                active[ events[evt_idx].index ] = events[evt_idx].entering;
                ++evt_idx;
            }

            Eigen::Vector3f pos = ray.origin + t * ray.direction;

            // get sigma_a, sigma_s for this segment
            float sigma_a, sigma_s;
            scene.evaluate_sigma(active, pos, sigma_a, sigma_s);
            float sigma_t = sigma_a + sigma_s;

            T *= std::exp(-sigma_t * step_size);
        }
        return T;
    }

public:
    PureRayMarching(const std::shared_ptr<Camera>& camera, float step_size = 0.01f, int env_samples = 20)
        : Integrator(camera), step_size(step_size), env_samples(env_samples) 
        {
            std::cout << "Medium-Agnostic Baseline Raymarcher" << std::endl;
        }

    void render(const Scene& scene, Image& image) override {
        const int W = (int)image.get_width();
        const int H = (int)image.get_height();

        #pragma omp parallel for collapse(2) schedule(dynamic,1)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float u = (x + 0.5f) / W;
                float v = (y + 0.5f) / H;
                Ray ray = camera->sample_ray({u, v});

                std::vector<PrimitiveHitEvent> primary_events;
                scene.intersect_events(ray, primary_events);
                if (primary_events.empty()) {
                    image.set_pixel(x, y, scene.env_color);
                    continue;
                }

                float t_end = primary_events.back().t;

                // setup primary marching loop
                std::vector<bool> primary_active(scene.get_num_primitives(), false);
                for (auto& e : primary_events) {
                    if (e.t == 0.0f && e.entering)
                        primary_active[e.index] = true;
                }

                float T = 1.0f;                        
                Eigen::Vector3f L(0.0f, 0.0f, 0.0f);   
                size_t idx_evt = 0, n_evt = primary_events.size();

                for (float t = 0.0f; t < t_end; t += step_size) {
                    // manage active primitives
                    while (idx_evt < n_evt && primary_events[idx_evt].t <= t) {
                        primary_active[ primary_events[idx_evt].index ] =
                            primary_events[idx_evt].entering;
                        ++idx_evt;
                    }

                    Eigen::Vector3f pos = ray.origin + t * ray.direction;

                    float sigma_a, sigma_s;
                    scene.evaluate_sigma(primary_active, pos, sigma_a, sigma_s);
                    float sigma_t = sigma_a + sigma_s;

                    if (sigma_s > 0.0f) {
                        // --- direct lights ---
                        Eigen::Vector3f Li(0.0f, 0.0f, 0.0f);
                        for (auto const& light : scene.lights) {
                            Eigen::Vector3f wi   = (light.position - pos).normalized();
                            float           dist = (light.position - pos).norm();
                            Ray   shadow_ray(pos, wi);

                            std::vector<PrimitiveHitEvent> shadow_events;
                            scene.intersect_events(shadow_ray, shadow_events);

                            std::vector<bool> shadow_active = primary_active;
                            for (size_t i = 0; i < shadow_active.size(); ++i) {
                                if (primary_active[i])
                                    shadow_events.insert(
                                        shadow_events.begin(),
                                        {0.0f, true, i}
                                    );
                            }

                            // march to get shadow transmittance
                            float Tr = march_transmittance(
                                scene,
                                shadow_ray,
                                dist,
                                shadow_events,
                                shadow_active,
                                step_size
                            );
                            Li += Tr * light.intensity / (dist*dist);
                        }

                        // --- environment ---
                        Eigen::Vector3f Le(0.0f, 0.0f, 0.0f);
                        for (int s = 0; s < env_samples; ++s) {
                            Eigen::Vector3f wi = sample_uniform_direction_old();
                            Ray   env_ray(pos, wi);

                            std::vector<PrimitiveHitEvent> env_events;
                            scene.intersect_events(env_ray, env_events);

                            std::vector<bool> env_active = primary_active;
                            for (size_t i = 0; i < env_active.size(); ++i) {
                                if (primary_active[i])
                                    env_events.insert(
                                        env_events.begin(),
                                        {0.0f, true, i}
                                    );
                            }

                            // march to get shadow transmittance
                            float t_env_end = env_events.empty()
                                              ? 0.0f
                                              : env_events.back().t;
                            float Tr_env = march_transmittance(
                                scene,
                                env_ray,
                                t_env_end,
                                env_events,
                                env_active,
                                step_size
                            );
                            Le += Tr_env * scene.env_color;
                        }
                        Le = (Le / float(env_samples)) * (4.0f * std::numbers::pi);

                        L += T * sigma_s * (Li + Le) * step_size * (1.0f / (4.0f * std::numbers::pi));
                    }

                    T *= std::exp(-sigma_t * step_size);
                }

                L += T * scene.env_color;

                image.set_pixel(x, y, L);
            }
        }
    }
};

// ============================================================================================
//  FREE-FLIGHT SAMPLING GAUSSIANS (SINGLE SCATTERING)
// ============================================================================================

class FreeFlightGaussians : public Integrator {
private:
    int num_samples;

public:
    FreeFlightGaussians(
        const std::shared_ptr<Camera>& cam,
        int   num_samples = 256
    ) : Integrator(cam), num_samples(num_samples) 
    {
        std::cout << "Gaussian free-flight renderer (single-scattering)" << std::endl;
        std::cout << "Distance Solver Method: "
        #if defined(UNIFORM)
            << "Uniform Approximation" << std::endl;
        #elif defined(BISECTION)
            << "Bisection" << std::endl;
        #elif defined(NEWTON)
            << "Newton-Raphson" << std::endl;
        #elif defined(ANALYTIC_PLUS_BISECTION)
            << "Analytic Inverse + Bisection" << std::endl;
        #elif defined(ANALYTIC_PLUS_NEWTON)
            << "Analytic Inverse + Newton-Raphson" << std::endl;
        #else
            #error "Must define a distance solver"
        #endif
    }

    void render(const Scene& scene, Image& image) override {
        const int W = image.get_width(), H = image.get_height();
        #pragma omp parallel for collapse(2) schedule(dynamic,1)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                Eigen::Vector3f pixel_L = Eigen::Vector3f::Zero();

                for (int si = 0; si < num_samples; ++si) {
                    uint64_t seed = derive_path_seed(x, y, si);
                    PCG32 rng(seed, 1);

                    int n = int(std::sqrt(num_samples)); // ASSUMING num_samples is power of 2
                    int sx = si % n;
                    int sy = si / n;

                    // stratified sample inside pixel
                    float u = (x + (sx + rng.uniform()) / n) / W;
                    float v = (y + (sy + rng.uniform()) / n) / H;
                    
                    Ray ray = camera->sample_ray({u,v});

                    const size_t Nprims = scene.get_num_primitives();

                    // thread local stuff
                    static thread_local std::vector<PrimitiveHitEvent> events_tls;
                    events_tls.reserve(std::max<size_t>(64, Nprims / 32) ); // tune reserve based on expected overlaps

                    scene.gmm->at(0).intersect_events(ray, events_tls);
                    if (events_tls.empty()) {
                        pixel_L += scene.env_color;
                        continue;
                    }

                    //  sample a target optical depth to get free-flight distance
                    float target_tau = -std::log(1.0f - rng.uniform());
                    std::vector<bool> active(scene.get_num_primitives(), false);
                    float acc_tau = 0.0f;
                    size_t ev_i = 0;
                    float t_prev = 0.0f, t_scatter = -1.0f;

                    // march over segments with fixed numbers of gaussians
                    std::vector<size_t> active_idxs;
                    while (ev_i < events_tls.size()) {
                        float t_evt = events_tls[ev_i].t;
                        active_idxs.clear();
                        for (size_t i=0;i<active.size();++i)
                            if (active[i]) active_idxs.push_back(i);
                        // accumulate depth til exceed target
                        float seg_tau = 0.0f;
                        for (auto idx : active_idxs)
                            seg_tau += scene.gmm->at(0).gaussians[idx]
                                         .optical_depth(ray, t_prev, t_evt);
                            if (acc_tau + seg_tau > target_tau) {
                                // exceeded! now find exact distance within segment
                                t_scatter = solve_distance(ray, t_prev, t_evt, active_idxs, target_tau - acc_tau, scene.gmm->at(0));
                                break;
                            }
                        acc_tau += seg_tau;
                        active[events_tls[ev_i].index] = events_tls[ev_i].entering;
                        t_prev = t_evt;
                        ++ev_i;
                    }

                    // if we never reached target, sample env
                    if (t_scatter < 0.0f) {
                        pixel_L += scene.env_color;
                        continue;
                    }

                    // evaluate medium at sampled distance
                    Eigen::Vector3f pos = ray.origin + t_scatter*ray.direction;
                    float albedo = scene.gmm->at(0).evaluate_albedo(active_idxs, pos);
                        
                    // sample single scatter inscattering from one random light (including env)
                    bool is_env = (rng.uniform() < 1.0f/(scene.lights.size()+1));
                    Eigen::Vector3f Li = Eigen::Vector3f::Zero();
                    if (!is_env) {
                        // pick random point light
                        int li = int(rng.uniform()*scene.lights.size());
                        const auto &L = scene.lights[li];
                        Eigen::Vector3f wi = (L.position - pos).normalized();
                        float dist = (L.position - pos).norm();

                        // get analytic Tr to light
                        Ray shadow(pos, wi);
                        float Tr = scene.gmm->at(0).transmittance_up_to(shadow, dist);
                        Li = Tr * L.intensity / (dist*dist);
                    } else {
                        // one env-sample: uniform dir
                        Eigen::Vector3f wi = sample_uniform_direction(rng);
                        Ray eray(pos, wi);
                        
                        float inf = std::numeric_limits<float>::infinity();
                        float Tr = scene.gmm->at(0).transmittance_up_to(eray, inf);
                        Li = Tr * scene.env_color * (4.0f * std::numbers::pi);
                    }

                    // apprpriately weight sample
                    float phase_pdf = 1.0f / (4.0f * std::numbers::pi);
                    float w_light = float(scene.lights.size() + 1);

                    pixel_L += (albedo * phase_pdf * w_light) * Li;
                }

                // average samples
                image.set_pixel(x, y, pixel_L / float(num_samples));
            }
        }
    }
};

// ============================================================================================
//  FREE-FLIGHT SAMPLING GAUSSIANS (MULTI SCATTERING)
// ============================================================================================

class MultiScatterGaussians : public Integrator {
private:
    int num_samples;
    int min_scatter;

    inline float get_free_flight_distance(
        const Ray& ray,
        const std::vector<PrimitiveHitEvent>& events,
        float target_tau,
        const GaussianMixtureModel& gmm,
        std::vector<size_t>& out_active_idxs,
        std::vector<int>& idx_pos,        // thread-local index->position
        std::vector<int>& idx_epoch,      // thread-local epoch marker per index
        int& epoch_counter                // thread-local epoch counter (incremented per-call)
    ) const
    {
        // start a new epoch
        int cur_epoch = ++epoch_counter;
        // handle very unlikely wrap around
        if (cur_epoch == 0) {
            std::fill(idx_epoch.begin(), idx_epoch.end(), 0);
            cur_epoch = ++epoch_counter;
        }

        double acc_tau = 0.0;
        float t_prev = 0.0f;
        size_t ev_i_local = 0;

        while (ev_i_local < events.size()) {
            float t_evt = events[ev_i_local].t;

            // compute optical depth contributed by currently-active gaussians for [t_prev, t_evt]
            double seg_tau = 0.0;
            for (size_t idx : out_active_idxs) {
                seg_tau += gmm.gaussians[idx].optical_depth(ray, t_prev, t_evt);
            }

            // if this segment would exceed target, solve inside it
            if (acc_tau + seg_tau > double(target_tau)) {
                float remaining_tau = float(double(target_tau) - acc_tau);
                float t_scatter = solve_distance(ray, t_prev, t_evt, out_active_idxs, remaining_tau, gmm);
                return t_scatter;
            }

            acc_tau += seg_tau;

            // update the active set for the NEXT segment
            uint32_t gidx = events[ev_i_local].index;
            bool entering = events[ev_i_local].entering;

            if (entering) {
                // add gidx if not already present in this epoch
                if (idx_epoch[gidx] != cur_epoch) {
                    idx_epoch[gidx] = cur_epoch;
                    idx_pos[gidx] = int(out_active_idxs.size());
                    out_active_idxs.push_back(gidx);
                }
            } else {
                // remove gidx (swap-remove idiom) only if it is present in this epoch
                if (idx_epoch[gidx] == cur_epoch) {
                    int pos = idx_pos[gidx];
                    // sanity check; pos should be valid if epoch matches
                    if (pos >= 0 && pos < (int)out_active_idxs.size()) {
                        size_t last_idx = out_active_idxs.back();
                        out_active_idxs[pos] = last_idx;
                        // update position of swapped-in element
                        idx_pos[last_idx] = pos;
                        out_active_idxs.pop_back();
                    }
                    // mark gidx as absent for this epoch
                    idx_epoch[gidx] = 0;
                }
            }

            // advance to next segment
            t_prev = t_evt;
            ++ev_i_local;
        }

        // no scatter found along events
        return -1.f;
    }

public:
    MultiScatterGaussians(
        const std::shared_ptr<Camera>& cam,
        int samples = 16,
        int min_bounces = 5
    ) : Integrator(cam), num_samples(samples), min_scatter(min_bounces) 
    {
        std::cout << "Gaussian free-flight renderer (multi-scattering)" << std::endl;
        std::cout << "Distance Solver Method: "
        #if defined(UNIFORM)
            << "UNIFORM Approximation" << std::endl;
        #elif defined(BISECTION)
            << "Bisection" << std::endl;
        #elif defined(NEWTON)
            << "Newton-Raphson" << std::endl;
        #elif defined(ANALYTIC_PLUS_BISECTION)
            << "Analytic Inverse + Bisection" << std::endl;
        #elif defined(ANALYTIC_PLUS_NEWTON)
            << "Analytic Inverse + Newton-Raphson" << std::endl;
        #else
            #error "Must define a distance solver"
        #endif
    }

    void render(const Scene& scene, Image& image) override {
        const int W = image.get_width(), H = image.get_height();
        #pragma omp parallel for collapse(2) schedule(dynamic,1)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                Eigen::Vector3f pixel_L = Eigen::Vector3f::Zero();
                const size_t Nprims = scene.get_num_primitives();

                for (int si = 0; si < num_samples; ++si) {
                    uint64_t seed = derive_path_seed(x, y, si);
                    PCG32 rng(seed, 1);
                    //if(x == 0 && y == 0) {std::cout << "seed: " << seed << std::endl; }

                    int n = int(std::sqrt(num_samples)); // ASSUMING num_samples is power of 2
                    int sx = si % n;
                    int sy = si / n;

                    // stratified sample inside pixel
                    float u = (x + (sx + rng.uniform()) / n) / W;
                    float v = (y + (sy + rng.uniform()) / n) / H;

                    Ray ray = camera->sample_ray({u, v});
                    
                    Eigen::Vector3f throughput = Eigen::Vector3f::Ones();
                    Eigen::Vector3f L_accum = Eigen::Vector3f::Zero();
                    bool alive = true;

                    // thread local stuff
                    // ======================================================================
                    static thread_local std::vector<PrimitiveHitEvent> events_tls;
                    events_tls.reserve(std::max<size_t>(64, Nprims / 8) ); // tune based on expected number of hits

                    static thread_local std::vector<size_t> active_idxs_tls;
                    active_idxs_tls.reserve(std::max<size_t>(16, Nprims / 128) ); // tune based on expected number of overlap

                    static thread_local std::vector<int> idx_pos_tls(  Nprims, 0 );    
                    static thread_local std::vector<int> idx_epoch_tls(  Nprims, 0 );
                    static thread_local int idx_epoch_counter = 0;
                    // ======================================================================

                    // multiple scattering loop
                    for (int bounce = 0; alive; ++bounce) {
                        // get intersection events
                        events_tls.clear();
                        scene.intersect_events(ray, events_tls);
                        if (events_tls.empty()) {
                            L_accum += throughput.cwiseProduct(scene.env_color);
                            break;
                        }

                        // sample a target optical depth
                        float target_tau = -std::log(1.0f - rng.uniform());
                        
                        active_idxs_tls.clear();

                        // get free-flight distance; fills active_idxs and returns t_scatter
                        float t_scatter = get_free_flight_distance(
                            ray,
                            events_tls,
                            target_tau,
                            scene.gmm->at(0),
                            active_idxs_tls,
                            idx_pos_tls,
                            idx_epoch_tls,
                            idx_epoch_counter
                        );

                        // if we never reached target, sample env
                        if (t_scatter < 0.f) {
                            L_accum += throughput.cwiseProduct(scene.env_color);
                            break;
                        }

                        // evaluate medium at sampled distance
                        Eigen::Vector3f pos = ray.origin + t_scatter * ray.direction;
                        float albedo = scene.gmm->at(0).evaluate_albedo(active_idxs_tls, pos);
                        
                        // Next Event Estimation: sample one light or env
                        bool is_env = (rng.uniform() < 1.f / (scene.lights.size() + 1));
                        Eigen::Vector3f Li = Eigen::Vector3f::Zero();

                        if (!is_env) {
                            // pick random point light
                            int li = int(rng.uniform()*scene.lights.size());
                            const auto &L = scene.lights[li];
                            Eigen::Vector3f wi = (L.position - pos).normalized();
                            float dist = (L.position - pos).norm();

                            // get analytic Tr to light
                            Ray shadow(pos, wi);
                            float Tr = scene.gmm->at(0).transmittance_up_to(shadow, dist);
                            Li = Tr * L.intensity / (dist * dist);
                        } else {
                            // one env-sample: uniform dir
                            Eigen::Vector3f wi = sample_uniform_direction(rng);
                            Ray eray(pos, wi);
                            float inf = std::numeric_limits<float>::infinity();
                            float Tr = scene.gmm->at(0).transmittance_up_to(eray, inf);
                            Li = Tr * scene.env_color * (4.0f * std::numbers::pi);
                        }

                        // appropriately weight NEE sample
                        float phase_pdf = 1.f / (4.0f * std::numbers::pi);
                        float w_ne = float(scene.lights.size() + 1);
                        Eigen::Vector3f contrib = (throughput * (albedo * phase_pdf * w_ne)).cwiseProduct(Li);
                        
                        L_accum += contrib;

                        // update throughput for recursive scattering sample
                        throughput *= albedo;

                        // russian roulette to stop multi-scatter after min_scatter
                        if (bounce >= min_scatter) {
                            float rr = std::min(throughput.maxCoeff(), 0.9f);
                            if (rng.uniform() > rr) break;
                            throughput /= rr;
                        }

                        // sample random scattering direction and continue
                        Eigen::Vector3f new_dir = sample_uniform_direction(rng);
                        ray = Ray(pos, new_dir);
                    }

                    pixel_L += L_accum;
                }

                // average samples
                image.set_pixel(x, y, pixel_L / float(num_samples));
            }
        }
    }
};

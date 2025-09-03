#pragma once

#include "integrator.h"
#include "optimizer.h"

#include <filesystem>
#include <unordered_set>

static inline void ensure_dir_exists(const std::string &path) {
    std::filesystem::path p(path);
    if (!std::filesystem::exists(p)) std::filesystem::create_directories(p);
}

static inline std::string out_filename(const std::string& dir, int it) {
    std::ostringstream ss;
    ss << dir << "/iter_" << std::setw(4) << std::setfill('0') << it << ".ppm";
    return ss.str();
}

static inline void compute_pixel_losses(const Image& I, const Image& Iref, std::vector<float>& out) {
    int W = I.get_width(), H = I.get_height();
    out.assign(W * H, 0.0f);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        Eigen::Vector3f a = I.get_pixel(x,y);
        Eigen::Vector3f b = Iref.get_pixel(x,y);
        Eigen::Vector3f d = a - b;
        out[y*W + x] = d.cwiseAbs().sum(); // L1 loss
    }
}

// ============================================================================================

// abstract class
class InverseIntegrator {
protected:
    const std::shared_ptr<Camera> camera;

public:
    InverseIntegrator(const std::shared_ptr<Camera> &camera) : camera(camera) {}
    virtual ~InverseIntegrator() = default;

    // run optimization on a scene
    virtual bool optimize(Scene scene_initial, const Image& I_ref) = 0;
};

// ============================================================================================
// stochastic finite difference inverse integrator
// ============================================================================================

#ifdef RECORD_PIXEL_GAUSSIANS

struct SFDDConfig {
    int max_iters = 1000;
    int save_every = 25;
    int num_stoch_samples = 4; // number of samples to estimate FD
    float lr = 1e-2f;
};

class StochasticFiniteDiffInverseIntegrator : public InverseIntegrator {
public:
    StochasticFiniteDiffInverseIntegrator(
        const std::shared_ptr<Camera>& cam,
        const std::shared_ptr<MultiScatterGaussians>& forward_integrator,
        const SFDDConfig& cfg = SFDDConfig()
    ) : InverseIntegrator(cam), forward_integrator(forward_integrator), cfg(cfg) {}

    bool optimize(Scene scene_initial, const Image& I_ref) override {
        using clock = std::chrono::high_resolution_clock;
        ensure_dir_exists("./sfd_output");

        if (!scene_initial.gmm || scene_initial.gmm->empty()) {
            std::cerr << "Scene has no GMM." << std::endl;
            return false;
        }

        // Base GMM (progressively optimized one)
        Scene scene_opt = scene_initial;
        scene_opt.gmm = std::vector<GaussianMixtureModel>{ scene_initial.gmm->at(0) };

        // Alt GMM (perturbed copy)
        Scene scene_alt = scene_initial;
        scene_alt.gmm = std::vector<GaussianMixtureModel>{ scene_initial.gmm->at(0) };

        GaussianMixtureModel& base_gmm = scene_opt.gmm->at(0);
        GaussianMixtureModel& alt_gmm  = scene_alt.gmm->at(0);

        // get params as a flat feature vector
        std::vector<float> params;
        base_gmm.pack_parameters(params);
        if (params.empty()) {
            std::cerr << "No params to optimize." << std::endl;
            return false;
        }

        // get eps for features
        std::vector<float> eps = make_default_eps_for_params(params);

        // Adam optimizer
        AdamOptimizer adam(params.size(), cfg.lr);

        // RNG
        std::mt19937 rng( std::random_device{}() );
        std::bernoulli_distribution coin(0.5);

        int W = I_ref.get_width(), H = I_ref.get_height();
        Image I_base(W,H), I_plus(W,H);
        std::vector<float> loss_base, loss_plus;

        auto t0 = clock::now();

        const size_t per = 11; // parameters per gaussian
        const size_t Ngauss = base_gmm.get_num_gaussians();

        for (int iter = 0; iter < cfg.max_iters; ++iter) {
            // -------------------------
            // 1) Render base and record per-pixel gaussian contributions
            // -------------------------
            std::vector<std::vector<uint32_t>> per_pixel_gaussians(W * H);
            forward_integrator->render(scene_opt, I_base, &per_pixel_gaussians);

            // base pixel-wise losses
            compute_pixel_losses(I_base, I_ref, loss_base);

            // invert per-pixel -> per-gaussian pixel lists
            std::vector<std::vector<int>> gaussian_pixels(Ngauss);
            for (int p = 0; p < W * H; ++p) {
                for (uint32_t g : per_pixel_gaussians[p]) {
                    if (g < Ngauss) gaussian_pixels[g].push_back(p);
                }
            }

            // -------------------------
            // 2) Gradient accumulator
            // -------------------------
            std::vector<double> param_grads_d(params.size(), 0.0);

            // -------------------------
            // 3) Stochastic sign vectors (forward difference)
            // -------------------------
            for (int sidx = 0; sidx < cfg.num_stoch_samples; ++sidx) {
                std::vector<int8_t> s(params.size());
                for (size_t i = 0; i < params.size(); ++i)
                    s[i] = coin(rng) ? 1 : -1;

                // build perturbed params
                std::vector<float> params_plus(params.size());
                for (size_t i = 0; i < params.size(); ++i)
                    params_plus[i] = params[i] + float(s[i]) * eps[i];

                // render perturbed and record per-pixel Gaussians
                apply_params_to_gmm_local(params_plus, alt_gmm);
                std::vector<std::vector<uint32_t>> per_pixel_gaussians_plus(W * H);
                forward_integrator->render(scene_alt, I_plus, &per_pixel_gaussians_plus);
                compute_pixel_losses(I_plus, I_ref, loss_plus);

                // invert for perturbed image
                std::vector<std::vector<int>> gaussian_pixels_plus(Ngauss);
                for (int p = 0; p < W * H; ++p) {
                    for (uint32_t g : per_pixel_gaussians_plus[p]) {
                        if (g < Ngauss) gaussian_pixels_plus[g].push_back(p);
                    }
                }

                // compute per-gaussian contributions F_plus - F_base
                std::vector<double> F_diff(Ngauss, 0.0);
                for (size_t g = 0; g < Ngauss; ++g) {
                    // union of base and perturbed pixel lists
                    std::unordered_set<int> union_pixels(
                        gaussian_pixels[g].begin(), gaussian_pixels[g].end());
                    union_pixels.insert(
                        gaussian_pixels_plus[g].begin(), gaussian_pixels_plus[g].end());

                    double Fp = 0.0, Fb = 0.0;
                    for (int p : union_pixels) {
                        Fp += double(loss_plus[p]);
                        Fb += double(loss_base[p]);
                    }
                    F_diff[g] = (Fp - Fb);
                }

                // accumulate gradient
                for (size_t i = 0; i < params.size(); ++i) {
                    size_t g = i / per;
                    if (g >= Ngauss) continue;
                    double denom = double(eps[i]);
                    if (std::abs(denom) < 1e-12) continue;
                    param_grads_d[i] += F_diff[g] * double(s[i]) / denom;
                }
            }

            // average
            for (size_t i = 0; i < params.size(); ++i)
                param_grads_d[i] /= double(cfg.num_stoch_samples);

            // convert to float for Adam
            std::vector<float> param_grads(params.size());
            for (size_t i = 0; i < params.size(); ++i)
                param_grads[i] = float(param_grads_d[i]);

            // update
            if (!adam.step(params, param_grads)) {
                std::cerr << "Adam step failed." << std::endl;
                return false;
            }

            // apply updated params
            apply_params_to_gmm_local(params, base_gmm);

            // save occasionally
            if (iter % cfg.save_every == 0) {
                std::string fn = out_filename("./sfd_output", iter);
                std::cout << "[SFD] iter " << iter << " saving " << fn << std::endl;
                forward_integrator->render(scene_opt, I_base);
                I_base.make_PPM(fn);

                compute_pixel_losses(I_base, I_ref, loss_base);
                double mean_loss = 0.0;
                for (float l : loss_base) mean_loss += l;
                mean_loss /= static_cast<double>(loss_base.size());
                std::cout << "[SFD] iter " << iter << " mean loss: " << mean_loss << std::endl;

                auto t1 = clock::now();
                std::chrono::duration<double> elapsed = t1 - t0;
                std::cout << "[SFD] iter " << iter << " elapsed "
                          << elapsed.count() << "s" << std::endl;
            }
        }

        // final save
        forward_integrator->set_num_samples(16384);
        forward_integrator->render(scene_opt, I_base);
        I_base.make_PPM(out_filename("./sfd_output", cfg.max_iters - 1));
        std::cout << "[SFD] Optimization finished." << std::endl;
        compute_pixel_losses(I_base, I_ref, loss_base);
        double mean_loss = 0.0;
        for (float l : loss_base) mean_loss += l;
        mean_loss /= static_cast<double>(loss_base.size());
        std::cout << "Final mean loss: " << mean_loss << std::endl;

        return true;
    }

private:
    std::shared_ptr<MultiScatterGaussians> forward_integrator;
    SFDDConfig cfg;
};

#endif
#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <optional>
#include <memory>

#include "gmm.h"
#include "smm.h"

struct Light {
    Eigen::Vector3f position;
    Eigen::Vector3f intensity; 
};

struct Scene {
    enum class VolumeType {
        GAUSSIANS,
        SPHERES,
        VOXELS
    } volume_type;

    std::optional<std::vector<GaussianMixtureModel>> gmm;
    std::optional<std::vector<SphereMixtureModel>> smm;
    //std::optional<std::shared_ptr<Voxels>> voxels;

    std::vector<Light> lights;
    Eigen::Vector3f env_color = {0.53f, 0.81f, 0.92f}; // default background radiance
    //Eigen::Vector3f env_color = {0.0f, 0.0f, 0.0f}; // default background radiance

    // =========================================================================================
    //  LOAD SCENES
    // =========================================================================================

    // Light:        l x y z  r g b
    // Sphere:       s x y z  radius sigma_a sigma_s
    static Scene load_SMM(const std::string& filename) {
        Scene s;
        s.volume_type = VolumeType::SPHERES;

        // load spheres into SMM...
            std::vector<Sphere> spheres;

        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open scene file: " + filename);
        }

        std::string tag;
        while (file >> tag) {
            if (tag == "l") {
                float x, y, z, r, g, b;
                file >> x >> y >> z >> r >> g >> b;
                s.lights.push_back({Eigen::Vector3f(x, y, z), Eigen::Vector3f(r, g, b)});
            }
            else if (tag == "s") {
                float x, y, z, radius, sigma_a, sigma_s;
                file >> x >> y >> z >> radius >> sigma_a >> sigma_s;
                spheres.emplace_back(Eigen::Vector3f(x, y, z), radius, sigma_a, sigma_s);
            }
        }

        s.smm = std::vector<SphereMixtureModel>{ SphereMixtureModel(spheres) };
        std::cout << "Loaded SMM scene from " << filename
                  << ", #spheres=" << spheres.size() << std::endl;
        return s;
    }

    // Light:        l x y z  r g b
    // Gaussian:     g x y z  cxx cxy cxz cyy cyz czz  density albedo [er eg eb]
    static Scene load_GMM(const std::string& filename) {
        Scene s;
        s.volume_type = VolumeType::GAUSSIANS;
        std::vector<Gaussian> gaussians;

        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open scene file: " + filename);
        }

        std::string tag;
        while (file >> tag) {
            if (tag == "l") {
                float x, y, z, r, g, b;
                file >> x >> y >> z >> r >> g >> b;
                s.lights.push_back({Eigen::Vector3f(x, y, z), Eigen::Vector3f(r, g, b)});
            }
            else if (tag == "g") {
                float mx, my, mz;
                float cxx, cxy, cxz, cyy, cyz, czz;
                float density, albedo;
                file >> mx >> my >> mz
                     >> cxx >> cxy >> cxz >> cyy >> cyz >> czz
                     >> density >> albedo;

                // try to read optional emission (three floats)
                Eigen::Vector3f emission(0.0f, 0.0f, 0.0f);
                int next = file.peek();
                if (next != '\n' && next != EOF) {
                    float er, eg, eb;
                    if (file >> er >> eg >> eb) {
                        emission = { er, eg, eb };
                    }
                }

                Eigen::Vector3f mean(mx, my, mz);
                Eigen::Matrix3f cov;
                cov << cxx, cxy, cxz,
                       cxy, cyy, cyz,
                       cxz, cyz, czz;

                gaussians.emplace_back(mean, cov, density, albedo, emission);
            }
        }

        // build GMM
        s.gmm = std::vector<GaussianMixtureModel>{ GaussianMixtureModel(gaussians) };
        std::cout << "Loaded GMM scene from " << filename
                  << ", #gaussians=" << gaussians.size() << std::endl;
        return s;
    }
    
    //static Scene load_VDB(const std::string& filename) { }

    // =========================================================================================
    //  CORE ROUTINES
    // =========================================================================================

    std::vector<PrimitiveHitEvent> intersect_events(const Ray& ray) const {
        switch (volume_type) {
            case VolumeType::SPHERES:
                if (smm && !smm->empty())
                    return smm->at(0).intersect_events(ray);
                throw std::runtime_error(
                    "Sphere mixture model not initialized");

            case VolumeType::GAUSSIANS:
                if (gmm && !gmm->empty())
                    return gmm->at(0).intersect_events(ray);
                throw std::runtime_error(
                    "Gaussian mixture model not initialized");

            case VolumeType::VOXELS:
                throw std::runtime_error("Voxel volume not supported");
            default:
                throw std::runtime_error("Trying to intersect with uninitialized medium");
        }
        // unreachable
        return {};
    }

    size_t get_num_primitives() const {
        switch (volume_type) {
            case VolumeType::SPHERES:
                if (smm && !smm->empty()) {
                    return (*smm)[0].get_num_spheres();
                }
                break;
            case VolumeType::GAUSSIANS:
                if (gmm && !gmm->empty())
                    return gmm->at(0).get_num_gaussians();
                break;
            default:
                throw std::runtime_error("Trying to get num_primitives with uninitialized medium");
        }
        return 0;
    }

    void evaluate_sigma(const std::vector<bool>& active, const Eigen::Vector3f& pos, float& sigma_a, float& sigma_s) const {
        switch(volume_type) {
            case VolumeType::SPHERES:
                if (smm && !smm->empty())
                    smm->at(0).evaluate_sigma(active, sigma_a, sigma_s);
                else {
                    sigma_a = 0.0f; sigma_s = 0.0f;
                }
                break;
            case VolumeType::GAUSSIANS:
                if (gmm && !gmm->empty()) {
                    gmm->at(0).evaluate_sigma(active, pos, sigma_a, sigma_s);
                }
                else {
                    sigma_a = 0.0f; sigma_s = 0.0f;
                }
                break;
            default:
                std::cerr << "Warning: getting sigma for uninitialized medium" << std::endl;
                sigma_a = 0.0f; sigma_s = 0.0f; 
        }
    }

    float transmittance_from_events(const Ray& ray, const std::vector<PrimitiveHitEvent>& events, float tmax) const {
        switch(volume_type) {
            case VolumeType::SPHERES:
                if (smm && !smm->empty())
                    return smm->at(0).transmittance_from_events(ray, events, tmax);
                else
                    return 1.0f;
            // Other volumes.
            default:
                std::cerr << "Warning: getting transmittance for uninitialized medium" << std::endl;
                return 1.0f;
        }
    }
};


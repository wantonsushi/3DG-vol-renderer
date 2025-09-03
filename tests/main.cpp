
#include "integrator.h"
#include "test_integrators.h"
#include "gif.h"

#include "inverse_integrator.h"

#include <iostream>
#include <memory>

//#define MAKE_GIF
//#define USE_INVERSE_OPT

using namespace std;


int main() {
    const unsigned int width = 512;
    const unsigned int height = 512;

    const Eigen::Vector3f camera_lookat{0, 1, 0};
    const float FOV = 0.25f * std::numbers::pi;

    // Load scene
    // Scene scene = Scene::load_SMM("../scenes/spheres/1_spheres.txt");
    Scene scene = Scene::load_GMM("../scenes/gaussians/2g_altered.txt");
    //Scene scene = Scene::load_GMM("../scenes/gaussians/250_random.txt");
    const int num_samples = 256;  // <-- must be power of 2 for stratified sampling
    const float step_size = 0.01f;

#ifndef MAKE_GIF
    Eigen::Vector3f camera_pos{0, 1, 6};
    Eigen::Vector3f view_dir = (camera_lookat - camera_pos).normalized();
    auto camera = make_shared<Pinhole_Camera>(camera_pos, view_dir, FOV);
    //auto camera = make_shared<Orthographic_Camera>(camera_pos, view_dir);

#ifndef USE_INVERSE_OPT
    // -------------------------------------------------------------------------
    // Forward rendering only (default)
    // -------------------------------------------------------------------------
    Image image(width, height);
    auto integrator = make_unique<MultiScatterGaussians>(camera, num_samples);

    auto start = chrono::high_resolution_clock::now();
    integrator->render(scene, image);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;
    cout << "Render time: " << elapsed.count() << " seconds" << endl;

    image.make_PPM("output.ppm");

#else
    // -------------------------------------------------------------------------
    // Inverse optimization (stochastic finite-diff)
    // -------------------------------------------------------------------------
    // Load reference image
    Image I_ref("2g_highspp.ppm");

    // Forward integrator for optimizer
    auto forward_integrator = make_shared<MultiScatterGaussians>(camera, num_samples);

    // Inverse optimizer
    SFDDConfig cfg;  // configure as needed
    auto inv_integrator = make_unique<StochasticFiniteDiffInverseIntegrator>(
        camera, forward_integrator, cfg
    );

    auto start = chrono::high_resolution_clock::now();
    inv_integrator->optimize(scene, I_ref);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;
    cout << "Inverse optimization time: " << elapsed.count() << " seconds" << endl;
#endif // USE_INVERSE_OPT

#else
    // -------------------------------------------------------------------------
    // GIF animation rendering mode
    // -------------------------------------------------------------------------
    const int num_frames = 120;
    const float radius = 6.0f;
    const float height_pos = 1.0f;
    const float fps = 30.0f;

    GifWriter gif;
    // delay is in hundredths of a second: 100/fps → centiseconds per frame
    GifBegin(&gif, "animation.gif", width, height, int(100.0f / fps));

    // Pre-allocate one RGBA buffer
    std::vector<uint8_t> frame_data(width * height * 4);

    for (int frame = 0; frame < num_frames; ++frame) {
        float angle = 2.0f * std::numbers::pi * (float(frame) / num_frames);
        Eigen::Vector3f camera_pos = camera_lookat + Eigen::Vector3f(
            radius * std::sin(angle),
            height_pos,
            radius * std::cos(angle)
        );

        Eigen::Vector3f view_dir = (camera_lookat - camera_pos).normalized();
        auto camera = make_shared<Orthographic_Camera>(camera_pos, view_dir);
        auto integrator = make_unique<RayMarchingGaussians>(camera);

        Image image(width, height);
        integrator->render(scene, image);
        auto frame_data = image.get_rgba_buffer();

        GifWriteFrame(&gif, frame_data.data(), width, height, int(100.0f/fps));
        cout << "Frame " << frame + 1 << " / " << num_frames << " complete.\n";
    }

    GifEnd(&gif);
    cout << "GIF saved." << endl;
#endif // MAKE_GIF

    return 0;
}
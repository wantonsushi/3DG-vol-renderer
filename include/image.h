#pragma once

#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>

class Image {
private:
    unsigned int width;
    unsigned int height;
    std::vector<float> pixels;

    unsigned int get_idx(unsigned int i, unsigned int j) const {
        return 3 * (j * width + i);
    }

public:
    Image(unsigned int width, unsigned int height) : width(width), height(height) {
        pixels.resize(3 * width * height);
    }

    Image(const std::string& filename) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) throw std::runtime_error("Failed to open PPM file: " + filename);

        std::string magic;
        infile >> magic;
        if (magic != "P6") throw std::runtime_error("Not a P6 PPM file.");

        infile >> width >> height;
        int maxval;
        infile >> maxval;
        infile.get(); // consume newline

        pixels.resize(3 * width * height);
        for (unsigned j = 0; j < height; ++j) {
            for (unsigned i = 0; i < width; ++i) {
                unsigned char rgb[3];
                infile.read(reinterpret_cast<char*>(rgb), 3);
                set_pixel(i, j, Eigen::Vector3f(rgb[0]/255.f, rgb[1]/255.f, rgb[2]/255.f));
            }
        }
    }

    unsigned int get_width() const { return width; }
    unsigned int get_height() const { return height; }

    Eigen::Vector3f get_pixel(unsigned int i, unsigned int j) const {
        const unsigned idx = get_idx(i, j);
        return Eigen::Vector3f(pixels[idx], pixels[idx + 1], pixels[idx + 2]);
    }

    void set_pixel(unsigned int i, unsigned int j, const Eigen::Vector3f& rgb) {
        const unsigned idx = get_idx(i, j);
        pixels[idx] = rgb[0];
        pixels[idx + 1] = rgb[1];
        pixels[idx + 2] = rgb[2];
    }

    void make_PPM(const std::string& filename) const {
        std::ofstream outfile(filename, std::ios::binary);

        auto clamp_255 = [](float val) -> unsigned char {
            return static_cast<unsigned char>(std::clamp((val * 255.0f), 0.0f, 255.0f));
        };

        // write PPM header
        outfile << "P6\n" << width << " " << height << "\n255\n";

        // write pixel data
        for (unsigned int j = 0; j < height; ++j) {
            for (unsigned int i = 0; i < width; ++i) {
                Eigen::Vector3f rgb = get_pixel(i, j);
                unsigned char r = clamp_255(rgb[0]);
                unsigned char g = clamp_255(rgb[1]);
                unsigned char b = clamp_255(rgb[2]);
                outfile << r << g << b;
            }
        }

        outfile.close();
    }

    // output a frame suitable for writing to the gif-maker
    std::vector<uint8_t> get_rgba_buffer() const {
        
        auto clamp_255 = [](float val) -> uint8_t {
            return static_cast<uint8_t>(std::clamp(val * 255.0f, 0.0f, 255.0f));
        };

        std::vector<uint8_t> buf(width * height * 4);
        for (unsigned j = 0; j < height; ++j) {
            for (unsigned i = 0; i < width; ++i) {
                Eigen::Vector3f rgb = get_pixel(i, j);
                size_t idx = 4 * (j * width + i);
                buf[idx + 0] = clamp_255(rgb[0]);  // R
                buf[idx + 1] = clamp_255(rgb[1]);  // G
                buf[idx + 2] = clamp_255(rgb[2]);  // B
                buf[idx + 3] = 255;                // A
            }
        }
        return buf;
    }
};


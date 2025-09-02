#pragma once

#include <omp.h>
#include <limits>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>

#include "gaussian.h"
#include "smm.h"

class GaussianMixtureModel {
private:
    struct BVHNode {
        Eigen::Vector3f bmin = Eigen::Vector3f::Constant( std::numeric_limits<float>::infinity() );
        Eigen::Vector3f bmax = Eigen::Vector3f::Constant( -std::numeric_limits<float>::infinity() );
        uint32_t leftFirst = 0; // left child index or first primitive index
        uint32_t count = 0;     // >0 => leaf with 'count' primitives
        bool isLeaf() const { return count > 0; }
    };

    std::vector<uint32_t> indices;   // permutation of gaussian indices
    std::vector<BVHNode> nodes;

    static float IntersectAABB(const Ray& ray, const Eigen::Vector3f& bmin, const Eigen::Vector3f& bmax) {
        Eigen::Vector3f rD = ray.direction.cwiseInverse();
        float tx1 = (bmin.x() - ray.origin.x()) * rD.x();
        float tx2 = (bmax.x() - ray.origin.x()) * rD.x();
        float tmin = std::min(tx1, tx2), tmax = std::max(tx1, tx2);
        float ty1 = (bmin.y() - ray.origin.y()) * rD.y();
        float ty2 = (bmax.y() - ray.origin.y()) * rD.y();
        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));
        float tz1 = (bmin.z() - ray.origin.z()) * rD.z();
        float tz2 = (bmax.z() - ray.origin.z()) * rD.z();
        tmin = std::max(tmin, std::min(tz1, tz2));
        tmax = std::min(tmax, std::max(tz1, tz2));
        if (tmax >= tmin && tmin < std::numeric_limits<float>::infinity() && tmax > 0.0f) return tmin;
        return std::numeric_limits<float>::infinity();
    }

    static inline bool point_in_aabb(const Eigen::Vector3f& p, const Eigen::Vector3f& bmin, const Eigen::Vector3f& bmax) {
        return (p.x() >= bmin.x() && p.x() <= bmax.x() &&
                p.y() >= bmin.y() && p.y() <= bmax.y() &&
                p.z() >= bmin.z() && p.z() <= bmax.z());
    }
public:
    std::vector<Gaussian> gaussians;

    GaussianMixtureModel() = default;

    explicit GaussianMixtureModel(const std::vector<Gaussian>& gs) 
        : gaussians(gs)
    {
        if (!gaussians.empty()) {

            auto start = std::chrono::high_resolution_clock::now();
            BuildBVH();
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "Gaussian BVH constructed: "
                      << gaussians.size() << " gaussians, "
                      << nodes.size() << " nodes" << std::endl;

            std::chrono::duration<double> elapsed = end - start;
            std::cout << "BVH build time: " << elapsed.count() << " seconds" << std::endl;
        }
    }

    size_t get_num_gaussians() const {
        return gaussians.size();
    }

    // evaluate absorption/scattering coefficients over multiple spatially varying gaussians    
    void evaluate_sigma(
        const std::vector<bool>& active,
        const Eigen::Vector3f&   pos,
        float&                   sigma_a,
        float&                   sigma_s
    ) const {
        float sum_mu_t       = 0.0f;
        float sum_mu_t_alb   = 0.0f;

        // 1) accumulate extinction and albedo
        for (size_t i = 0; i < gaussians.size(); ++i) {
            if (!active[i]) continue;
            float mu_t_i = gaussians[i].mu_t(pos);
            sum_mu_t     += mu_t_i;
            sum_mu_t_alb += mu_t_i * gaussians[i].get_albedo();
        }

        if (sum_mu_t <= 0.0f) {
            sigma_s = sigma_a = 0.0f;
            return;
        }

        // 2) absorption coefficient
        float a_mix = sum_mu_t_alb / sum_mu_t;

        // 3) scattering coefficient
        sigma_s = a_mix       * sum_mu_t;
        sigma_a = (1.0f - a_mix) * sum_mu_t;
    }

    float evaluate_albedo(
        const std::vector<size_t>& active_idxs,
        const Eigen::Vector3f&     pos
    ) const {
        float sum_mu_t     = 0.0f;
        float sum_mu_t_alb = 0.0f;

        for (size_t idx : active_idxs) {
            float mu_t_i = gaussians[idx].mu_t(pos);
            sum_mu_t     += mu_t_i;
            sum_mu_t_alb += mu_t_i * gaussians[idx].get_albedo();
        }

        float a = sum_mu_t_alb / sum_mu_t;
        return std::clamp(a, 0.0f, 1.0f);
    }

    // get transmittance over all active gaussians = exp(-sum(optical depths))
    float transmittance_over_segment(
        const Ray& ray,
        float t0,
        float t1,
        const std::vector<size_t>& active_indices
    ) const {
        float optical_depth_sum = 0.0f;
        for (size_t i : active_indices) {
            optical_depth_sum += gaussians[i].optical_depth(ray, t0, t1);
        }
        return std::exp(-optical_depth_sum);
    }

    // =========================================================================================
    // =========================================================================================
    #define USE_BVH
    //#define SAH  // <-- sucks!
    // =========================================================================================
    // =========================================================================================

    inline void intersect_events(
        const Ray& ray,
        std::vector<PrimitiveHitEvent>& out_events
    ) const {
    #ifdef USE_BVH
        intersect_events_BVH(ray, out_events);
    #else
        intersect_events_naive(ray, out_events);
    #endif
    }
    
    inline float transmittance_up_to(const Ray &ray, float tmax) const {
    #ifdef USE_BVH
        return transmittance_up_to_BVH(ray, tmax);
    #else
        return transmittance_up_to_naive(ray, tmax);
    #endif
    }


    // =========================================================================================
    // Naive intersection routines
    
    // get all individual intersection events, sorted in ascending order
    inline void intersect_events_naive(
        const Ray& ray,
        std::vector<PrimitiveHitEvent>& out_events
    ) const {
        for (size_t i = 0; i < gaussians.size(); ++i) {
            float t0, t1;
            //if (gaussians[i].intersect_whitening(ray, t0, t1)) {
            if (gaussians[i].intersect_direct(ray, t0, t1)) {
                if (t0 >= 0.0f) out_events.push_back({ t0, true,  i });
                if (t1 >= 0.0f) out_events.push_back({ t1, false, i });
            }
        }
        std::sort(out_events.begin(), out_events.end());
    }

    // much faster transmittance along ray from t=0 to t = tmax
    // no sorting, for single-scatter shadow transmittance/next-event-estimation
    inline float transmittance_up_to_naive(const Ray &ray, float tmax) const {
        if (tmax <= 0.0f) return 1.0f;

        double optical_depth_sum = 0.0; // use double for accumulate safety
        for (size_t i = 0; i < gaussians.size(); ++i) {
            float g_t0, g_t1;
            // only consider this gaussian if it intersects the ray
            if (!gaussians[i].intersect_direct(ray, g_t0, g_t1))
                continue;

            // clip to [0, tmax]
            float a = std::max(0.0f, g_t0);
            float b = std::min(tmax, g_t1);
            if (b > a) {
                optical_depth_sum += gaussians[i].optical_depth(ray, a, b);
            }
        }

        return std::exp(-float(optical_depth_sum));
    }

    // =========================================================================================
    // BVH utils

    void BuildBVH() {
        const uint32_t N = static_cast<uint32_t>(gaussians.size());
        indices.resize(N);
        for (uint32_t i = 0; i < N; ++i) indices[i] = i;

        // precompute per-gaussian AABBs
        std::vector<Eigen::Vector3f> gmins(N), gmaxs(N);

        for (int i = 0; i < (int)N; ++i) {
            gaussians[i].get_aabb(gmins[i], gmaxs[i]);
        }

        nodes.clear();
        nodes.reserve(std::max<size_t>(1, 2 * N));
        nodes.emplace_back(); // root

        nodes[0].leftFirst = 0;
        nodes[0].count = N;

        // compute root bounds
        nodes[0].bmin = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
        nodes[0].bmax = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
        for (uint32_t i = 0; i < N; ++i) {
            nodes[0].bmin = nodes[0].bmin.cwiseMin(gmins[i]);
            nodes[0].bmax = nodes[0].bmax.cwiseMax(gmaxs[i]);
        }

        // recursively subdivide
        SubdivideNode(0);
    }

    void UpdateNodeBounds(uint32_t nodeIdx) {
        BVHNode& node = nodes[nodeIdx];
        node.bmin = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
        node.bmax = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
        uint32_t first = node.leftFirst;
        for (uint32_t i = 0; i < node.count; ++i) {
            uint32_t gidx = indices[first + i];
            Eigen::Vector3f gmin, gmax;
            gaussians[gidx].get_aabb(gmin, gmax);
            node.bmin = node.bmin.cwiseMin(gmin);
            node.bmax = node.bmax.cwiseMax(gmax);
        }
    }

    // Evaluate SAH using child AABB surface areas × primitive counts
    float EvaluateSAH(const BVHNode& node, int axis, float pos) const {
        Eigen::Vector3f leftMin  = Eigen::Vector3f::Constant( std::numeric_limits<float>::infinity() );
        Eigen::Vector3f leftMax  = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity() );
        Eigen::Vector3f rightMin = Eigen::Vector3f::Constant( std::numeric_limits<float>::infinity() );
        Eigen::Vector3f rightMax = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity() );


        uint32_t leftCount = 0, rightCount = 0;
        uint32_t first = node.leftFirst;

        for (uint32_t i = 0; i < node.count; ++i) {
            uint32_t gidx = indices[first + i];
            Eigen::Vector3f gmin, gmax;
            gaussians[gidx].get_aabb(gmin, gmax);

            if (gaussians[gidx].centroid()[axis] < pos) {
                leftMin  = leftMin.cwiseMin(gmin);
                leftMax  = leftMax.cwiseMax(gmax);
                ++leftCount;
            } else {
                rightMin = rightMin.cwiseMin(gmin);
                rightMax = rightMax.cwiseMax(gmax);
                ++rightCount;
            }
        }

        if (leftCount == 0 || rightCount == 0)
            return std::numeric_limits<float>::infinity();

        auto surfaceArea = [](const Eigen::Vector3f& bmin, const Eigen::Vector3f& bmax) {
            Eigen::Vector3f d = (bmax - bmin).cwiseMax(0.0f);
            return 2.0f * (d.x() * d.y() + d.x() * d.z() + d.y() * d.z());
        };

        float leftArea  = surfaceArea(leftMin, leftMax);
        float rightArea = surfaceArea(rightMin, rightMax);

        return leftArea * leftCount + rightArea * rightCount; // SAH cost
    }

    // Full SubdivideNode that switches between SAH and midpoint split
    void SubdivideNode(uint32_t nodeIdx) {
        BVHNode& node = nodes[nodeIdx];
        if (node.count <= 4) return; // leaf threshold

    #ifdef SAH
        uint32_t first = node.leftFirst;
        uint32_t count = node.count;

        int bestAxis = -1;
        uint32_t bestSplit = 0;
        float bestCost = std::numeric_limits<float>::infinity();

        // Lambda to compute SAH surface area
        auto surfaceArea = [](const Eigen::Vector3f& bmin, const Eigen::Vector3f& bmax) {
            Eigen::Vector3f d = (bmax - bmin).cwiseMax(0.0f);
            return 2.0f * (d.x() * d.y() + d.x() * d.z() + d.y() * d.z());
        };

        // Precompute parent cost
        Eigen::Vector3f parentMin = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
        Eigen::Vector3f parentMax = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t gidx = indices[first + i];
            Eigen::Vector3f gmin, gmax;
            gaussians[gidx].get_aabb(gmin, gmax);
            parentMin = parentMin.cwiseMin(gmin);
            parentMax = parentMax.cwiseMax(gmax);
        }
        float parentCost = surfaceArea(parentMin, parentMax) * count;

        // Try each axis
        for (int axis = 0; axis < 3; ++axis) {
            // --- 1. Sort by centroid along axis ---
            std::sort(indices.begin() + first, indices.begin() + first + count,
                [&](uint32_t a, uint32_t b) {
                    return gaussians[a].centroid()[axis] < gaussians[b].centroid()[axis];
                });

            // --- 2. Prefix bounds ---
            std::vector<Eigen::Vector3f> leftMin(count), leftMax(count);
            Eigen::Vector3f curMin = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
            Eigen::Vector3f curMax = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
            for (uint32_t i = 0; i < count; ++i) {
                Eigen::Vector3f gmin, gmax;
                gaussians[indices[first + i]].get_aabb(gmin, gmax);
                curMin = curMin.cwiseMin(gmin);
                curMax = curMax.cwiseMax(gmax);
                leftMin[i] = curMin;
                leftMax[i] = curMax;
            }

            // --- 3. Suffix bounds ---
            std::vector<Eigen::Vector3f> rightMin(count), rightMax(count);
            curMin = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
            curMax = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
            for (int i = count - 1; i >= 0; --i) {
                Eigen::Vector3f gmin, gmax;
                gaussians[indices[first + i]].get_aabb(gmin, gmax);
                curMin = curMin.cwiseMin(gmin);
                curMax = curMax.cwiseMax(gmax);
                rightMin[i] = curMin;
                rightMax[i] = curMax;
            }

            // --- 4. Evaluate splits ---
            for (uint32_t i = 1; i < count; ++i) { // split after i-1
                float leftArea  = surfaceArea(leftMin[i - 1], leftMax[i - 1]);
                float rightArea = surfaceArea(rightMin[i], rightMax[i]);
                float cost = leftArea * i + rightArea * (count - i);

                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplit = i;
                }
            }
        }

        // Abort if not beneficial
        if (bestAxis < 0 || bestCost >= parentCost) return;

        // --- 5. Re-sort along best axis and partition ---
        std::sort(indices.begin() + first, indices.begin() + first + count,
            [&](uint32_t a, uint32_t b) {
                return gaussians[a].centroid()[bestAxis] < gaussians[b].centroid()[bestAxis];
            });

        uint32_t leftCount = bestSplit;
        if (leftCount == 0 || leftCount == count) return; // failed partition

    #else
        // midpoint fallback (unchanged)
        uint32_t first = node.leftFirst;
        Eigen::Vector3f e = node.bmax - node.bmin;
        int axis = (e.y() > e.x()) ? 1 : 0;
        if (e.z() > e[axis]) axis = 2;
        float splitPos = 0.5f * (node.bmin[axis] + node.bmax[axis]);

        uint32_t i = node.leftFirst;
        uint32_t j = i + node.count - 1;
        while (i <= j) {
            uint32_t gidx = indices[i];
            if (gaussians[gidx].centroid()[axis] < splitPos) ++i;
            else { std::swap(indices[i], indices[j]); --j; }
        }
        uint32_t leftCount = i - node.leftFirst;
        if (leftCount == 0 || leftCount == node.count) return;
    #endif

        // --- Create children (shared with both modes) ---
        uint32_t leftChild = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();
        uint32_t rightChild = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();

        nodes[leftChild].leftFirst = node.leftFirst;
        nodes[leftChild].count = leftCount;
        nodes[rightChild].leftFirst = node.leftFirst + leftCount;
        nodes[rightChild].count = node.count - leftCount;

        node.leftFirst = leftChild;
        node.count = 0;

        UpdateNodeBounds(leftChild);
        UpdateNodeBounds(rightChild);

        SubdivideNode(leftChild);
        SubdivideNode(rightChild);
    }

    // =========================================================================================
    // intersection routines using BVH

    // use the same stack
    static inline std::vector<int>& thread_local_stack() {
        thread_local std::vector<int> stack;
        return stack;
    }

    inline void intersect_events_BVH(
        const Ray& ray,
        std::vector<PrimitiveHitEvent>& out_events
    ) const {
        out_events.clear();
        if (gaussians.empty() || nodes.empty()) return;

        auto& stack = thread_local_stack();
        stack.clear();
        stack.reserve(std::max<size_t>(64, nodes.size() / 16));
        stack.push_back(0); // root

        while (!stack.empty()) {
            int nodeIdx = stack.back();
            stack.pop_back();
            const BVHNode& node = nodes[nodeIdx];

            float tmin = IntersectAABB(ray, node.bmin, node.bmax);
            if (tmin == std::numeric_limits<float>::infinity()) continue;

            if (node.isLeaf()) {
                // push leaf events directly into out_events (avoid temporaries)
                for (int ii = 0; ii < (int)node.count; ++ii) {
                    uint32_t gidx = indices[node.leftFirst + ii];
                    float t0, t1;
                    if (gaussians[gidx].intersect_direct(ray, t0, t1)) {
                        if (t0 >= 0.0f) out_events.emplace_back(PrimitiveHitEvent{ t0, true,  gidx });
                        if (t1 >= 0.0f) out_events.emplace_back(PrimitiveHitEvent{ t1, false, gidx });
                    }
                }
            } else {
                // ordered traversal: compute child entry distances and push farther first
                int leftIdx  = static_cast<int>(node.leftFirst);
                int rightIdx = static_cast<int>(node.leftFirst + 1);

                const BVHNode& leftNode  = nodes[leftIdx];
                const BVHNode& rightNode = nodes[rightIdx];

                float dl = IntersectAABB(ray, leftNode.bmin, leftNode.bmax);
                float dr = IntersectAABB(ray, rightNode.bmin, rightNode.bmax);

                // push farther first so nearer is processed next
                if (dl > dr) {
                    if (dl != std::numeric_limits<float>::infinity()) stack.push_back(leftIdx);
                    if (dr != std::numeric_limits<float>::infinity()) stack.push_back(rightIdx);
                } else {
                    if (dr != std::numeric_limits<float>::infinity()) stack.push_back(rightIdx);
                    if (dl != std::numeric_limits<float>::infinity()) stack.push_back(leftIdx);
                }
            }
        }

        // sort events by distance (t). Use lambda to be robust in case PrimitiveHitEvent lacks operator<
        if (!out_events.empty()) {
            std::sort(out_events.begin(), out_events.end(), [](const PrimitiveHitEvent& a, const PrimitiveHitEvent& b){
                return a.t < b.t;
            });
        }
    }

    inline float transmittance_up_to_BVH(const Ray &ray, float tmax) const {
        if (tmax <= 0.0f) return 1.0f;
        if (gaussians.empty() || nodes.empty()) return 1.0f;

        double optical_depth_sum = 0.0; // accumulate in double for accuracy

        auto& stack = thread_local_stack();
        stack.clear();
        stack.reserve(std::max<size_t>(64, nodes.size() / 16));
        stack.push_back(0); // root

        while (!stack.empty()) {
            int nodeIdx = stack.back();
            stack.pop_back();
            const BVHNode& node = nodes[nodeIdx];

            float tmin = IntersectAABB(ray, node.bmin, node.bmax);

            // Skip if ray misses AABB OR if AABB starts beyond tmax
            if (tmin == std::numeric_limits<float>::infinity() || tmin > tmax)
                continue;

            if (node.isLeaf()) {
                for (int ii = 0; ii < (int)node.count; ++ii) {
                    uint32_t gidx = indices[node.leftFirst + ii];
                    float g_t0, g_t1;
                    if (!gaussians[gidx].intersect_direct(ray, g_t0, g_t1)) continue;

                    // clip to [0, tmax]
                    float a = std::max(0.0f, g_t0);
                    float b = std::min(tmax, g_t1);
                    if (b > a) {
                        optical_depth_sum += gaussians[gidx].optical_depth(ray, a, b);
                    }
                }
            } else {
                int leftIdx  = static_cast<int>(node.leftFirst);
                int rightIdx = static_cast<int>(node.leftFirst + 1);

                const BVHNode& leftNode  = nodes[leftIdx];
                const BVHNode& rightNode = nodes[rightIdx];

                float dl = IntersectAABB(ray, leftNode.bmin, leftNode.bmax);
                float dr = IntersectAABB(ray, rightNode.bmin, rightNode.bmax);

                // Skip children completely outside [0, tmax]
                if (dl == std::numeric_limits<float>::infinity() || dl > tmax) dl = std::numeric_limits<float>::infinity();
                if (dr == std::numeric_limits<float>::infinity() || dr > tmax) dr = std::numeric_limits<float>::infinity();

                // Traverse nearer child first
                if (dl > dr) {
                    if (dl != std::numeric_limits<float>::infinity()) stack.push_back(leftIdx);
                    if (dr != std::numeric_limits<float>::infinity()) stack.push_back(rightIdx);
                } else {
                    if (dr != std::numeric_limits<float>::infinity()) stack.push_back(rightIdx);
                    if (dl != std::numeric_limits<float>::infinity()) stack.push_back(leftIdx);
                }
            }
        }

        return std::exp(-float(optical_depth_sum));
    }
};
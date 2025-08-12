import numpy as np
import argparse
import os

def random_rotation_matrix():
    """Generate a random 3D rotation matrix."""
    rand = np.random.normal(size=(3, 3))
    Q, _ = np.linalg.qr(rand)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def generate_covariance_matrix(diameters, rotation):
    """Compute covariance matrix from axis-aligned diameters and rotation."""
    variances = (np.array(diameters) / 2) ** 2
    scaled = np.diag(variances)
    cov = rotation @ scaled @ rotation.T
    return cov

def generate_gaussian():
    """Generate a single Gaussian entry."""
    x = np.random.uniform(-1.0, 1.0)
    y = np.random.uniform(0.0, 2.0)
    z = np.random.uniform(-1.0, 1.0)

    dx = np.random.uniform(0.1, 0.35)
    dy = np.random.uniform(0.1, 0.35)
    dz = np.random.uniform(0.1, 0.35)

    R = random_rotation_matrix()
    cov = generate_covariance_matrix([dx, dy, dz], R)

    cxx = cov[0, 0]
    cxy = cov[0, 1]
    cxz = cov[0, 2]
    cyy = cov[1, 1]
    cyz = cov[1, 2]
    czz = cov[2, 2]

    density = np.random.uniform(0.2, 0.5)
    albedo = np.random.uniform(0.5, 1.0)
    er = np.random.uniform(0.0, 1.0)
    eg = np.random.uniform(0.0, 1.0)
    eb = np.random.uniform(0.0, 1.0)

    return f"g {x:.4f} {y:.4f} {z:.4f}  {cxx:.6f} {cxy:.6f} {cxz:.6f} {cyy:.6f} {cyz:.6f} {czz:.6f}  {density:.4f} {albedo:.4f}  {er:.4f} {eg:.4f} {eb:.4f}"

def generate_scene(num_gaussians, output_filename):
    """Generate the full scene file and save to ../scenes/gaussians/<filename>."""
    target_dir = os.path.join("..", "scenes", "gaussians")
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, output_filename)

    lines = ["// Scene with random Gaussians"]
    for _ in range(num_gaussians):
        lines.append(generate_gaussian())

    with open(full_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote scene to {full_path} with {num_gaussians} Gaussians.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Gaussian scene file.")
    parser.add_argument("num_gaussians", type=int, help="Number of Gaussians to generate.")
    parser.add_argument("filename", type=str, help="Output file name (e.g., test.txt).")
    args = parser.parse_args()

    generate_scene(args.num_gaussians, args.filename)

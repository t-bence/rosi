import argparse
import csv
import math


def generate_cylindrical_csv(filename, num_points, radius_R, height_z):
    """
    Generates a CSV with x = R*cos(phi), y = R*sin(phi), z = constant.
    phi is equidistant between 0 and 2*PI.
    """
    # Calculate step size for phi to cover a full circle (0 to 2*pi)
    step_phi = (2 * math.pi) / num_points

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["x", "y", "z"])

        for i in range(num_points):
            phi = i * step_phi
            x = radius_R * math.cos(phi)
            y = radius_R * math.sin(phi)
            z = height_z

            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{z}"])

    print(f"CSV file '{filename}' created successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV of points on a circle at constant height."
    )

    # Define arguments
    parser.add_argument(
        "-N", type=int, default=100, help="Number of points (default: 100)"
    )
    parser.add_argument("-R", type=float, default=5.0, help="Radius R (default: 5.0)")
    parser.add_argument(
        "-Z", type=float, default=10.0, help="Constant height Z (default: 10.0)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="mics.csv", help="Output CSV filename"
    )

    args = parser.parse_args()

    generate_cylindrical_csv(args.output, args.N, args.R, args.Z)


if __name__ == "__main__":
    main()

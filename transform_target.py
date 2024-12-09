import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

# Load the flipped point cloud from a PLY file
flipped_ply_file_path = "data/artec3d/flipped_model_spanner.ply"  # Replace with your flipped PLY file path
flipped_point_cloud = PyntCloud.from_file(flipped_ply_file_path)

# Access the point cloud data as a DataFrame
points = flipped_point_cloud.points

# Add Gaussian noise to x, y, z coordinates
mean = 0  # Mean of the Gaussian distribution
std_dev = 0.5  # Standard deviation of the Gaussian distribution

# Generate noise for each coordinate
noise_x = np.random.normal(mean, std_dev, points.shape[0])
noise_y = np.random.normal(mean, std_dev, points.shape[0])
noise_z = np.random.normal(mean, std_dev, points.shape[0])

# Add noise to the points
points["x"] += noise_x
points["y"] += noise_y
points["z"] += noise_z

# Save the noisy point cloud to a new PLY file
noisy_ply_file_path = "data/artec3d/noisy_flipped_model_spanner.ply"  # Output file path
noisy_point_cloud = PyntCloud(points)
noisy_point_cloud.to_file(noisy_ply_file_path)

print(f"Noisy point cloud saved to: {noisy_ply_file_path}")



# import numpy as np
# import pandas as pd
# from pyntcloud import PyntCloud

# # Function to generate a random 3D rotation matrix
# def generate_random_rotation_matrix():
#     # Generate random angles for x, y, z axes
#     random_angles = np.random.uniform(0, 2 * np.pi, 3)
    
#     # Rotation matrices for each axis
#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(random_angles[0]), -np.sin(random_angles[0])],
#         [0, np.sin(random_angles[0]), np.cos(random_angles[0])]
#     ])

#     Ry = np.array([
#         [np.cos(random_angles[1]), 0, np.sin(random_angles[1])],
#         [0, 1, 0],
#         [-np.sin(random_angles[1]), 0, np.cos(random_angles[1])]
#     ])

#     Rz = np.array([
#         [np.cos(random_angles[2]), -np.sin(random_angles[2]), 0],
#         [np.sin(random_angles[2]), np.cos(random_angles[2]), 0],
#         [0, 0, 1]
#     ])

#     # Combine rotations: R = Rz * Ry * Rx
#     rotation_matrix = Rz @ Ry @ Rx
#     return rotation_matrix

# # Load the point cloud from a PLY file
# ply_file_path = "data/artec3d/model_spanner.ply"  # Replace with your PLY file path
# point_cloud = PyntCloud.from_file(ply_file_path)

# # Access the point cloud data as a DataFrame
# points = point_cloud.points

# # Extract the x, y, z coordinates as a NumPy array
# xyz = points[['x', 'y', 'z']].to_numpy()

# # Generate a random rotation matrix
# rotation_matrix = generate_random_rotation_matrix()

# # Apply the random rotation to the point cloud
# rotated_xyz = xyz @ rotation_matrix.T

# # Update the DataFrame with the rotated coordinates
# points[['x', 'y', 'z']] = rotated_xyz

# # Save the rotated point cloud to a new PLY file
# rotated_ply_file_path = "data/artec3d/rotated_model_spanner.ply"  # Output file path
# rotated_point_cloud = PyntCloud(points)
# rotated_point_cloud.to_file(rotated_ply_file_path)

# print(f"Rotated point cloud saved to: {rotated_ply_file_path}")

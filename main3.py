import laspy
import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

# Step 1: Define feature and residual calculation functions
def fit_plane(points):
    """Fit a plane to points in 3D using PCA and return the normal and centroid."""
    pca = PCA(n_components=3)
    pca.fit(points)
    normal_vector = pca.components_[0]  # Normal vector of the plane (first principal component)
    centroid = np.mean(points, axis=0)  # Centroid of the points
    return normal_vector, centroid

def calculate_orthogonal_distances(points, normal_vector, centroid):
    """Calculate the orthogonal distances of points to a plane defined by a normal and centroid."""
    distances = np.abs((points - centroid).dot(normal_vector))  # Orthogonal distance to the plane
    return distances

def calculate_residual(distances):
    """Calculate the residual value as the root mean square of orthogonal distances."""
    residual = np.sqrt(np.mean(distances ** 2))  # Root mean square of distances
    return residual

# Step 2: Load the .las file and create a DataFrame with point data
file_path = "data/cloud_final_kopi.las"

with laspy.open(file_path) as f:
    las = f.read()
    points = np.vstack((las.X * las.header.scale[0] + las.header.offset[0],
                        las.Y * las.header.scale[1] + las.header.offset[1],
                        las.Z * las.header.scale[2] + las.header.offset[2])).T
    classifications = las.classification
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['classification'] = classifications

# Step 3: Voxelize and prepare for feature calculation
voxel_size = 1.0  # Adjust based on desired voxel resolution
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# Step 4: Calculate residual and classification-based color for each voxel
voxel_features = {}

# Define a color map for classifications
unique_classes = df['classification'].unique()
colormap = plt.colormaps.get_cmap("viridis")  # Only specify the colormap name

# Map each class to a specific color
class_to_color = {cls: colormap(i / len(unique_classes))[:3] for i, cls in enumerate(unique_classes)}
# Create colored voxel meshes
voxel_meshes = []
for voxel in voxel_grid.get_voxels():
    # Calculate the voxel bounds and center
    voxel_center = voxel.grid_index
    min_bound = np.array(voxel_grid.origin) + np.array(voxel_center) * voxel_size
    max_bound = min_bound + voxel_size
    voxel_center_coords = min_bound + voxel_size / 2

    # Filter points within this voxel
    points_in_voxel = df[(df['x'] >= min_bound[0]) & (df['x'] < max_bound[0]) &
                         (df['y'] >= min_bound[1]) & (df['y'] < max_bound[1]) &
                         (df['z'] >= min_bound[2]) & (df['z'] < max_bound[2])][['x', 'y', 'z']].values

    # Ensure there are enough points for meaningful plane fitting (at least 3)
    if len(points_in_voxel) >= 3:
        # Step 1: Fit a plane using PCA (fitting to voxel points)
        normal_vector, centroid = fit_plane(points_in_voxel)

        # Step 2: Calculate orthogonal distances of points to the fitted plane
        distances = calculate_orthogonal_distances(points_in_voxel, normal_vector, centroid)

        # Step 3: Calculate the residual value as the root mean square of distances
        residual = calculate_residual(distances)
        
        # Add residual and other features to the dictionary for each voxel
        voxel_features[tuple(voxel_center)] = {"residual": residual}

        # Determine the most common classification in the voxel and assign a color
        common_class = Counter(df['classification']).most_common(1)[0][0]
        color = class_to_color[common_class]

        # Create a voxel cube and color it
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        cube.translate(voxel_center_coords - np.array([voxel_size / 2, voxel_size / 2, voxel_size / 2]))
        cube.paint_uniform_color(color)  # Color the cube based on classification

        voxel_meshes.append(cube)

# Step 5: Combine all voxel meshes and visualize
voxel_mesh_combined = o3d.geometry.TriangleMesh()
for mesh in voxel_meshes:
    voxel_mesh_combined += mesh

# Visualize the color-coded voxel grid
o3d.visualization.draw_geometries([voxel_mesh_combined], window_name="Color-Coded Voxel Grid with Residuals", width=800, height=600)

# Convert residuals and other features to a DataFrame
features_df = pd.DataFrame.from_dict(voxel_features, orient='index')
print(features_df.head())

# Optional: Save the features to a CSV file
features_df.to_csv("voxel_features_with_residual.csv", index_label="voxel_id")

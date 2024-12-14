import laspy
import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load point cloud and preprocess
def load_point_cloud(file_path):
    with laspy.open(file_path) as f:
        las = f.read()
        
        # Create point cloud data
        data = {
            'x': las.X * las.header.scale[0] + las.header.offset[0],
            'y': las.Y * las.header.scale[1] + las.header.offset[1],
            'z': las.Z * las.header.scale[2] + las.header.offset[2],
            'intensity': las.intensity,
            'classification': las.classification
        }
        df = pd.DataFrame(data)
    return df

# Convert DataFrame to Open3D PointCloud
def df_to_o3d_point_cloud(df):
    points = df[['x', 'y', 'z']].values
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# Step 1: Voxelization of the Point Cloud
def voxelize_point_cloud(pcd, voxel_size=1.0):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return voxel_grid

# Step 2 & 3: Feature Selection and Calculation
def calculate_features(df, voxel_grid):
    # Extract voxel-based features: mean, std dev of intensity within each voxel
    df['voxel_id'] = voxel_grid.get_voxel(df[['x', 'y', 'z']].values)
    voxel_features = df.groupby('voxel_id').agg({
        'intensity': ['mean', 'std'],
        'x': 'mean', 'y': 'mean', 'z': 'mean'
    }).fillna(0)
    
    # Flatten the MultiIndex columns
    voxel_features.columns = ['_'.join(col) for col in voxel_features.columns]
    return voxel_features

# Step 4: Classification
def classify_voxel_features(features, labels):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Initialize and train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate classifier
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return clf

# Step 5: Manual Evaluation (optional)
def manual_evaluation(df, voxel_grid, classifier):
    # Predict classifications for the full data
    features = calculate_features(df, voxel_grid)
    predictions = classifier.predict(features)

    # Add predictions to original data for comparison
    df['predicted_classification'] = predictions[df['voxel_id']]
    return df

# Main program
def main():
    file_path = "data/cloud_final_kopi.las"
    
    # Load point cloud
    df = load_point_cloud(file_path)
    
    # Convert to Open3D PointCloud and perform voxelization
    pcd = df_to_o3d_point_cloud(df)
    voxel_size = 1.0
    voxel_grid = voxelize_point_cloud(pcd, voxel_size)

    # Feature selection and calculation
    features = calculate_features(df, voxel_grid)
    labels = df.groupby('voxel_id')['classification'].first()

    # Classification
    classifier = classify_voxel_features(features, labels)

    # Optional: Manual evaluation
    evaluated_df = manual_evaluation(df, voxel_grid, classifier)
    print("Sample of manually evaluated data:\n", evaluated_df.head())

if __name__ == "__main__":
    main()
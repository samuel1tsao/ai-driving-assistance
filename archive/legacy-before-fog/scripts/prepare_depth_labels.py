import os
import sys
import subprocess
import tarfile
from pathlib import Path
import csv
import numpy as np
from PIL import Image

def main():
    root_dir = Path(r"c:\Users\tsao1\OneDrive\Documents\workspace\Ai Driving Assistance app")
    downloads_dir = Path(r"C:\Users\tsao1\Downloads")
    datasets_dir = root_dir / "data" / "depth" / "datasets" / "nuscenes"
    depth_out_dir = root_dir / "data" / "depth" / "datasets" / "nuscenes_depth" / "val"
    manifest_path = root_dir / "data" / "depth" / "manifest.csv"

    datasets_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)


    print("Extracting nuScenes archives...")
    archives = [
        "v1.0-trainval_meta.tgz",
        "v1.0-trainval01_blobs.tgz",
        "v1.0-trainval02_blobs.tgz",
        "v1.0-trainval04_blobs.tgz",
        "v1.0-trainval08_blobs.tgz",
        "v1.0-trainval09_blobs.tgz",
        "v1.0-trainval10_blobs.tgz"
    ]
    
    if (datasets_dir / "v1.0-trainval").exists():
        print("Dataset already extracted. Skipping extraction.")
    else:
        for arc in archives:
            arc_path = downloads_dir / arc
            if not arc_path.exists():
                print(f"Warning: Archive {arc} not found in Downloads, skipping.")
                continue
            print(f"Extracting {arc}...")
            # Use Windows native tar for speed
            subprocess.run(["tar", "-xf", str(arc_path), "-C", str(datasets_dir)], check=True)

    print("Extraction complete. Initializing NuScenes API...")
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(datasets_dir), verbose=True)
    val_scenes = create_splits_scenes()['val']

    print("Projecting LiDAR to CAM_FRONT and saving depth maps...")
    
    manifest_rows = []
    
    for scene in nusc.scene:
        if scene['name'] not in val_scenes:
            continue
            
        current_sample_token = scene['first_sample_token']
        while current_sample_token != '':
            sample = nusc.get('sample', current_sample_token)
            
            cam_token = sample['data']['CAM_FRONT']
            lidar_token = sample['data']['LIDAR_TOP']
            
            cam_data = nusc.get('sample_data', cam_token)
            cam_filename = cam_data['filename']
            
            # Map pointcloud to image
            try:
                points, coloring, im = nusc.explorer.map_pointcloud_to_image(lidar_token, cam_token)
            except Exception as e:
                print(f"Error projecting sample {current_sample_token}: {e}")
                current_sample_token = sample['next']
                continue
                
            # Create a 2D depth map
            width, height = im.size
            depth_map = np.zeros((height, width), dtype=np.float32)
            
            # Sort points by depth descending (so closer points overwrite further points)
            sort_idx = np.argsort(-coloring)
            points = points[:, sort_idx]
            coloring = coloring[sort_idx]
            
            # Filter bounds
            x = np.round(points[0, :]).astype(int)
            y = np.round(points[1, :]).astype(int)
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (coloring > 0)
            
            x = x[valid]
            y = y[valid]
            z = coloring[valid]
            
            depth_map[y, x] = z
            
            # Save depth map
            out_filename = Path(cam_filename).stem + ".npy"
            out_path = depth_out_dir / out_filename
            np.save(out_path, depth_map)
            
            # Add to manifest
            rel_image = Path("datasets/nuscenes") / cam_filename
            rel_depth = Path("datasets/nuscenes_depth/val") / out_filename
            
            manifest_rows.append({
                "image": str(rel_image).replace("\\", "/"),
                "depth": str(rel_depth).replace("\\", "/"),
                "depth_scale": "1.0",
                "split": "val"
            })
            
            current_sample_token = sample['next']

    print(f"Writing manifest with {len(manifest_rows)} entries...")
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "depth", "depth_scale", "split"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
            
    print("nuScenes depth preparation complete!")

if __name__ == "__main__":
    main()

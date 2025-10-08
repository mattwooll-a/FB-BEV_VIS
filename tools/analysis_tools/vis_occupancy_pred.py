#!/usr/bin/env python
"""
Create video from sequence of occupancy predictions
Processes folder of .npy/.npz files and creates video/gif
"""

import numpy as np
np.bool = np.bool_
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from pathlib import Path
from tqdm import tqdm
import subprocess

point_cloud_range = [-50, -50, -2, 50, 50, 5]
voxel_size = [0.2, 0.2, 0.2]

def get_class_colors():
    """Define colors for each class"""
    classname_to_color = {
        'ignore_class': (0, 0, 0),
        'barrier': (112, 128, 144),
        'bicycle': (220, 20, 60),
        'bus': (255, 127, 80),
        'car': (255, 158, 0),
        'construction_vehicle': (233, 150, 70),
        'motorcycle': (255, 61, 99),
        'pedestrian': (0, 0, 230),
        'traffic_cone': (47, 79, 79),
        'trailer': (255, 140, 0),
        'truck': (255, 99, 71),
        'driveable_surface': (0, 207, 191),
        'other_flat': (175, 0, 75),
        'sidewalk': (75, 0, 75),
        'terrain': (112, 180, 60),
        'manmade': (222, 184, 135),
        'vegetation': (0, 175, 0)
    }
    
    colors = {}
    for i, (name, rgb) in enumerate(classname_to_color.items()):
        colors[i] = tuple(c/255.0 for c in rgb)
    
    return colors, list(classname_to_color.keys())

def get_grid_coords(dims, resolution):
    """Generate grid coordinates for voxels"""
    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid

def load_voxel_data(filepath):
    """Load voxel data from .npy or .npz file"""
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        if 'pred' in data:
            return data['pred']
        elif 'semantics' in data:
            return data['semantics']
        elif 'arr_0' in data:
            return data['arr_0']
        else:
            # Return first array found
            return data[data.files[0]]
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def render_frame(voxels, frame_idx, total_frames, voxel_size=0.2, 
                downsample=1, view_angle=(30, 45), ax_limits=None, fixed_z_range=(0, 3),
                z_layers=None, hide_classes=None):
    """
    Render a single frame
    Args:
        z_layers: If specified, only render voxels in these Z-layer indices (e.g., [0] for ground layer)
        hide_classes: List of class indices to hide from rendering (e.g., [0] to hide ignore_class)
    Returns: fig, ax, ax_limits
    """
    
    # Get grid coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )
    
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    occupied_voxels = grid_coords[
        (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)
    ]
    
    # Debug: print unique classes found (only first frame)
    if frame_idx == 0:
        unique_vals = np.unique(occupied_voxels[:, 3].astype(int))
        print(f"  Unique class IDs found: {unique_vals}")
    
    # Filter out hidden classes
    if hide_classes is not None:
        mask = ~np.isin(occupied_voxels[:, 3].astype(int), hide_classes)
        occupied_voxels = occupied_voxels[mask]
    
    # Filter by Z layers if specified
    if z_layers is not None:
        # Convert Z coordinates back to layer indices
        z_indices = np.round((occupied_voxels[:, 2] - point_cloud_range[2]) / voxel_size).astype(int)
        mask = np.isin(z_indices, z_layers)
        occupied_voxels = occupied_voxels[mask]
        if frame_idx == 0:  # Only print once
            print(f"  Filtering to Z-layers: {z_layers}")
    
    # Downsample if requested
    if downsample > 1 and len(occupied_voxels) > 0:
        indices = np.random.choice(len(occupied_voxels), 
                                  max(1, len(occupied_voxels)//downsample), 
                                  replace=False)
        occupied_voxels = occupied_voxels[indices]
    
    # Get colors
    class_colors, class_names = get_class_colors()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    if len(occupied_voxels) > 0:
        x = occupied_voxels[:, 0]
        y = occupied_voxels[:, 1]
        z = occupied_voxels[:, 2]
        classes = occupied_voxels[:, 3].astype(int)
        
        # Map colors - use grey (0.5, 0.5, 0.5) for unknown classes
        colors = np.array([class_colors.get(c, (0.5, 0.5, 0.5)) for c in classes])
        
        # Count unknown classes
        unknown_classes = [c for c in classes if c not in class_colors]
        if unknown_classes and frame_idx == 0:
            unique_unknown = np.unique(unknown_classes)
            print(f"  WARNING: Found {len(unknown_classes)} voxels with unknown class IDs: {unique_unknown}")
            print(f"  These will appear as grey. Consider hiding them with --hide-classes {','.join(map(str, unique_unknown))}")
        
        ax.scatter(x, y, z, c=colors, marker='s', s=20, alpha=0.8, edgecolors='none')
        
        # Calculate limits for consistent view across frames
        if ax_limits is None:
            # Use X and Y range, but fix Z range
            max_range_xy = max(x.max()-x.min(), y.max()-y.min()) / 2.0
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            ax_limits = {
                'xlim': (mid_x - max_range_xy, mid_x + max_range_xy),
                'ylim': (mid_y - max_range_xy, mid_y + max_range_xy),
                'zlim': fixed_z_range
            }
    else:
        # Empty frame - use default limits
        if ax_limits is None:
            ax_limits = {
                'xlim': (-50, 50),
                'ylim': (-50, 50),
                'zlim': fixed_z_range
            }
    
    # Set labels and limits
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title(f'Occupancy Prediction - Frame {frame_idx+1}/{total_frames}', 
                fontsize=12, pad=15)
    
    ax.set_xlim(ax_limits['xlim'])
    ax.set_ylim(ax_limits['ylim'])
    ax.set_zlim(ax_limits['zlim'])
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    

    
    # Set aspect ratio to prevent squashing
    ax.set_box_aspect([1, 1, 0.1])  # Make Z axis visually shorter
    
    plt.tight_layout()
        # Add legend only for visible classes
    if len(occupied_voxels) > 0:
        unique_classes = np.unique(classes)
        legend_elements = []
        for cls in unique_classes:
            if cls < len(class_names):
                color = class_colors.get(cls, (0.5, 0.5, 0.5))
                legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                                 markerfacecolor=color, markersize=10,
                                                 label=class_names[cls]))
        
        if legend_elements and len(legend_elements) <= 20:
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(1.05, 1), fontsize=9)
    return fig, ax, ax_limits

def natural_sort_key(s):
    """
    Natural sorting key function to handle numeric strings properly
    Converts '10' to be sorted after '2' instead of before it
    """
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def find_files(input_path):
    """Find all .npy and .npz files in directory with natural sorting"""
    files = []
    
    if os.path.isfile(input_path):
        return [input_path]
    
    if os.path.isdir(input_path):
        # Look for .npy files first
        npy_files = glob.glob(os.path.join(input_path, "**/*.npy"), recursive=True)
        npz_files = glob.glob(os.path.join(input_path, "**/*.npz"), recursive=True)
        files = npy_files + npz_files
        
        if not files:
            raise ValueError(f"No .npy or .npz files found in {input_path}")
        
        # Sort using natural sorting (handles numbers correctly)
        files = sorted(files, key=natural_sort_key, reverse=True)
        
        print(f"First few files (in order):")
        for f in files[:5]:
            print(f"  {os.path.basename(f)}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")
    
    return files

def create_video(input_path, output_file='occupancy_video.mp4', 
                fps=10, downsample=1, view_angle=(30, 45),
                rotate=False, max_frames=None, z_range=(0, 3), z_layers=None,
                hide_classes=17):
    
    print(f"Searching for files in {input_path}...")
    files = find_files(input_path)
    
    if max_frames:
        files = files[:max_frames]
    
    print(f"Found {len(files)} files to process")
    
    if len(files) == 0:
        raise ValueError("No files found!")
    
    # Create temporary directory for frames
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    
    print(f"Rendering frames to {temp_dir}...")
    print(f"Z-axis fixed to range: {z_range}")
    if z_layers is not None:
        print(f"Rendering only Z-layers: {z_layers}")
    if hide_classes is not None:
        print(f"Hiding classes: {hide_classes}")
    
    ax_limits = None  # Keep consistent across frames
    
    for i, filepath in enumerate(tqdm(files, desc="Rendering frames")):
        try:
            voxels = load_voxel_data(filepath)
            
            # Calculate rotation if enabled
            if rotate:
                azim = view_angle[1] + (360.0 / len(files)) * i
                current_view = (view_angle[0], azim)
            else:
                current_view = view_angle
            
            fig, ax, ax_limits = render_frame(
                voxels, i, len(files), 
                voxel_size=0.2,
                downsample=downsample,
                view_angle=current_view,
                ax_limits=ax_limits,
                fixed_z_range=z_range,
                z_layers=z_layers,
                hide_classes=hide_classes
            )
            
            # Save frame
            frame_path = temp_dir / f"frame_{i:05d}.png"
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Failed to process {filepath}: {e}")
            continue
    
    print(f"Creating video: {output_file}")
    
    # Check output format
    if output_file.endswith('.gif'):
        # Create GIF using imageio
        try:
            import imageio
            frames = []
            for frame_file in sorted(temp_dir.glob("frame_*.png")):
                frames.append(imageio.imread(frame_file))
            imageio.mimsave(output_file, frames, fps=fps)
            print(f"✓ GIF saved to {output_file}")
        except ImportError:
            print("Error: imageio not installed. Install with: pip install imageio")
            print("Falling back to MP4 format...")
            output_file = output_file.replace('.gif', '.mp4')
    
    if output_file.endswith('.mp4'):
        # Create MP4 using ffmpeg
        frame_pattern = str(temp_dir / "frame_%05d.png")
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            output_file
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Video saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating video with ffmpeg: {e}")
            print("Make sure ffmpeg is installed: sudo apt-get install ffmpeg")
            print(f"Frames are saved in {temp_dir} if you want to create video manually")
        except FileNotFoundError:
            print("Error: ffmpeg not found. Install with: sudo apt-get install ffmpeg")
            print(f"Frames are saved in {temp_dir} if you want to create video manually")
    
    # Option to keep or delete frames
    print(f"\nFrames saved in {temp_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Create video from sequence of occupancy predictions'
    )
    parser.add_argument('input', help='Path to folder with .npy/.npz files')
    parser.add_argument('--output', '-o', default='occupancy_video.mp4',
                       help='Output video filename (.mp4 or .gif)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')
    parser.add_argument('--downsample', type=int, default=1,
                       help='Voxel downsample factor (default: 1)')
    parser.add_argument('--elevation', type=float, default=30,
                       help='Camera elevation angle in degrees (default: 30)')
    parser.add_argument('--azimuth', type=float, default=45,
                       help='Camera azimuth angle in degrees (default: 45)')
    parser.add_argument('--rotate', action='store_true',
                       help='Rotate camera around scene over time')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--z-min', type=float, default=0,
                       help='Minimum Z-axis value (default: 0)')
    parser.add_argument('--z-max', type=float, default=3,
                       help='Maximum Z-axis value (default: 5)')
    parser.add_argument('--z-layers', type=str, default=None,
                       help='Comma-separated Z-layer indices to render (e.g., "0" for ground, "0,1,2" for bottom 3 layers)')
    parser.add_argument('--hide-classes', type=str, default=None,
                       help='Comma-separated class indices to hide (e.g., "0" to hide ignore_class)')
    parser.add_argument('--hide-ignore', action='store_true',
                       help='Shortcut to hide ignore_class (class 0)')
    
    args = parser.parse_args()

    z_layers = [10,11,12,13,14,15,16,17,18]
    if args.z_layers:
        z_layers = [int(x.strip()) for x in args.z_layers.split(',')]
    
    hide_classes = 17
    if args.hide_ignore:
        hide_classes = [0]
    elif args.hide_classes:
        hide_classes = [int(x.strip()) for x in args.hide_classes.split(',')]
    
    create_video(
        args.input,
        args.output,
        fps=args.fps,
        downsample=args.downsample,
        view_angle=(args.elevation, args.azimuth),
        rotate=args.rotate,
        max_frames=args.max_frames,
        z_range=(args.z_min, args.z_max),
        z_layers=z_layers,
        hide_classes=hide_classes
    )

if __name__ == "__main__":
    main()
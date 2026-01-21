#!/usr/bin/env python3
"""
Script to test frame filtering using DINO embeddings and clustering.

This script loads simulation frames from the cache directory and clusters
them based on visual similarity using DINO embeddings.
"""

from untitled.filter.embedder import DINOEmbedder
from untitled.filter.filter import cluster_frames
from untitled.simulator.simulation_data import (
    SimulationData,
    SimulationFrameData,
    load_metadata,
)

import argparse
import sys
from pathlib import Path


SIMULATION_IMAGE_LOG_FPS = 4


def load_simulation_data(run_dir: Path) -> SimulationData:
    """
    Load simulation data from a run directory.
    
    Args:
        run_dir: Path to the run directory containing images/ and metadata/ subdirs.
    
    Returns:
        SimulationData object with all frames loaded.
    """
    images_dir = run_dir / "images"
    metadata_dir = run_dir / "metadata"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")
    
    # Get all image files sorted by name
    image_files = sorted(images_dir.glob("*.png"))
    
    frames = []
    for image_file in image_files:
        # Find corresponding metadata file
        metadata_file = metadata_dir / f"{image_file.stem}.json"
        if metadata_file.exists():
            metadata = load_metadata(metadata_file)
            frame = SimulationFrameData(
                image_name=image_file.name,
                metadata=metadata
            )
            frames.append(frame)
        else:
            print(f"Warning: No metadata found for {image_file.name}, skipping.")
    
    return SimulationData(
        image_dir=images_dir,
        frame_count=len(frames),
        frames=frames
    )


def undersample_frames_per_minute(
    simulation_data: SimulationData, max_frames_per_minute: int
) -> SimulationData:
    """
    Undersample frames to keep at most max_frames_per_minute per minute of simulation.
    
    Args:
        simulation_data: The simulation data to undersample.
        max_frames_per_minute: Maximum frames to keep per minute.
    
    Returns:
        A new SimulationData with undersampled frames.
    """
    frames_per_minute = 60 * SIMULATION_IMAGE_LOG_FPS
    step = max(1, frames_per_minute // max_frames_per_minute)
    sampled_frames = [f for i, f in enumerate(simulation_data.frames) if i % step == 0]
    
    return SimulationData(
        image_dir=simulation_data.image_dir,
        frame_count=len(sampled_frames),
        frames=sampled_frames
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test frame filtering using DINO embeddings and clustering."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).parent.parent / "cache" / "sce_005" / "run_000",
        help="Path to the run directory (default: cache/sce_005/run_000)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for clustering (default: 0.9)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)"
    )
    parser.add_argument(
        "--max-frames-per-minute",
        type=int,
        default=None,
        help="Maximum number of frames to keep per minute of simulation time (default: all)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information about each cluster"
    )
    
    args = parser.parse_args()
    
    print(f"Loading simulation data from: {args.run_dir}")
    
    # Load simulation data
    try:
        simulation_data = load_simulation_data(args.run_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Optionally limit number of frames
    if args.max_frames is not None and args.max_frames < simulation_data.frame_count:
        simulation_data.frames = simulation_data.frames[:args.max_frames]
        simulation_data.frame_count = len(simulation_data.frames)
    
    # Optionally undersample frames per minute
    if args.max_frames_per_minute is not None and simulation_data.frame_count > 0:
        original_count = simulation_data.frame_count
        simulation_data = undersample_frames_per_minute(
            simulation_data, args.max_frames_per_minute
        )
        print(f"Undersampled from {original_count} to {simulation_data.frame_count} frames "
              f"(max {args.max_frames_per_minute} frames/minute)")
    
    print(f"Loaded {simulation_data.frame_count} frames")
    print(f"Using similarity threshold: {args.threshold}")
    print()
    
    # Initialize embedder
    print("Initializing DINO embedder (this may take a moment to download the model)...")
    embedder = DINOEmbedder()
    print("Embedder initialized.")
    print()
    
    # Perform clustering
    print("Clustering frames...")
    clusters = cluster_frames(simulation_data, embedder, args.threshold)
    print(f"Created {len(clusters)} clusters from {simulation_data.frame_count} frames")
    print()
    
    # Print cluster statistics
    print("=" * 60)
    print("CLUSTER STATISTICS")
    print("=" * 60)
    
    cluster_sizes = [c.get_data().frame_count for c in clusters]
    
    print(f"Total clusters: {len(clusters)}")
    print(f"Total frames: {sum(cluster_sizes)}")
    print(f"Average cluster size: {sum(cluster_sizes) / len(clusters):.2f}")
    print(f"Min cluster size: {min(cluster_sizes)}")
    print(f"Max cluster size: {max(cluster_sizes)}")
    print()
    
    # Print cluster size distribution
    print("Cluster size distribution:")
    size_counts = {}
    for size in cluster_sizes:
        size_counts[size] = size_counts.get(size, 0) + 1
    for size in sorted(size_counts.keys()):
        print(f"  Size {size}: {size_counts[size]} cluster(s)")
    print()
    
    # Print detailed cluster info if verbose
    if args.verbose:
        print("=" * 60)
        print("CLUSTER DETAILS")
        print("=" * 60)
        for i, cluster in enumerate(clusters):
            data = cluster.get_data()
            print(f"\nCluster {i + 1} ({data.frame_count} frames):")
            for frame in data.frames:
                print(f"  - {frame.image_name} (step {frame.metadata.step})")
    
    # Print summary of unique frames (first frame from each cluster)
    print("=" * 60)
    print("REPRESENTATIVE FRAMES (first frame from each cluster)")
    print("=" * 60)
    for i, cluster in enumerate(clusters):
        data = cluster.get_data()
        first_frame = data.frames[0]
        print(f"Cluster {i + 1:3d}: {first_frame.image_name} (step {first_frame.metadata.step:5d}, cluster size: {data.frame_count})")


if __name__ == "__main__":
    main()

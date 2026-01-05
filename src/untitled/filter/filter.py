from untitled.core.base import LOG
from untitled.filter.embedder import Embedder
from untitled.simulator.simulation_data import SimulationData, SimulationFrameData

from typing import List
from pathlib import Path
import torch


class Cluster:
    """
    Represents a cluster of similar simulation frames based on image embeddings.
    """

    def __init__(self, image_dir: Path, embedder: Embedder):
        # Attributes
        self._data: SimulationData = SimulationData(
            image_dir=image_dir,
            frame_count=0,
            frames=[]               # List of SimulationFrameData
        )
        self._centroid: torch.Tensor | None = None      # Centroid representation
        self._centroid_weight: int = 0                  # Number of frames contributing to the centroid
        self._embedder: Embedder = embedder             # Embedder to use for embeddings

        # Cache
        self._last_calculated_frame: SimulationFrameData | None = None
        self._last_calculated_embedding: torch.Tensor | None = None

    def _embed(self, frame: SimulationFrameData) -> torch.Tensor:
        return self._embedder.embed(self._data.image_dir / frame.image_name)


    def calculate_similarity(self, frame: SimulationFrameData) -> float | None:
        """
        Calculate similarity score between the cluster and a given frame.

        Note that the provided frame should be in the same image directory as the cluster frames.

        Args:
            frame (SimulationFrameData): The frame to compare against the cluster.

        Returns a float similarity score (cosine similarity) or None if the centroid is not defined.
        """
        # If the centroid was never computed, it means that the cluster is empty
        if self._centroid is None:
            return None
        
        # Compute embedding for the given frame
        frame_embedding = self._embed(frame)
        frame_embedding = torch.nn.functional.normalize(frame_embedding, dim=0)

        # Update cache
        self._last_calculated_frame = frame
        self._last_calculated_embedding = frame_embedding

        # Compute cosine similarity between the centroid and the frame embedding
        similarity_score = torch.nn.functional.cosine_similarity(self._centroid, frame_embedding, dim=0).item()
        return similarity_score

    def add(self, frame: SimulationFrameData):
        """
        Add a frame to the cluster.

        Args:
            frame (SimulationFrameData): The frame to add to the cluster.
        """
        # Add frame to the cluster data
        self._data.frames.append(frame)
        self._data.frame_count += 1

        # If frame is not in cache, calculate its embedding
        frame_embedding: torch.Tensor = None
        if frame != self._last_calculated_frame or self._last_calculated_embedding is None:
            frame_embedding = self._embed(frame)
            frame_embedding = torch.nn.functional.normalize(frame_embedding, dim=0)
            self._last_calculated_frame = frame
            self._last_calculated_embedding = frame_embedding
        else:
            # Use cached embedding to avoid redundant computation
            frame_embedding = self._last_calculated_embedding

        # Update centroid using new embedding
        total_weight = self._centroid_weight + 1
        if self._centroid is None:
            self._centroid = frame_embedding
        else:
            self._centroid = (self._centroid * self._centroid_weight + frame_embedding) / total_weight
        self._centroid = torch.nn.functional.normalize(self._centroid, dim=0)
        self._centroid_weight = total_weight    # Update weight to avoid redundant recomputations

    def get_data(self) -> SimulationData:
        """
        Get the list of frames in the cluster.
        """
        return self._data


def cluster_frames(frame_data: SimulationData, embedder: Embedder, similarity_threshold: float, max_cluster_count: int = 0) -> List[Cluster]:
    """
    Cluster frames based on similarity threshold.
    """
    # Skip if there are no frames
    if not frame_data.frames:
        return []

    # Creat first cluster with first frame
    clusters: List[Cluster] = []
    current_cluster: Cluster = Cluster(frame_data.image_dir, embedder)
    current_cluster.add(frame_data.frames[0])
    clusters.append(current_cluster)

    # For each frame, check if is similar enough to the current cluster
    # If yes, add to that cluster, else create a new cluster, which will become the current cluster
    for i in range(1, len(frame_data.frames)):
        frame = frame_data.frames[i]
        similarity = current_cluster.calculate_similarity(frame)
        if similarity >= similarity_threshold:
            # Add to current cluster
            current_cluster.add(frame)
        else:
            # Create a new cluster
            current_cluster = Cluster(frame_data.image_dir, embedder)
            current_cluster.add(frame)
            clusters.append(current_cluster)

            # If we reached the maximum number of clusters,
            # we can just add the remaining frames to the last cluster
            if max_cluster_count != 0 and len(clusters) >= max_cluster_count:
                for j in range(i + 1, len(frame_data.frames)):
                    current_cluster.add(frame_data.frames[j])

                # Display a user warning
                LOG.warning(f"Reached maximum cluster count of {max_cluster_count} at frame index {i}. "
                      f"The {len(frame_data.frames) - (i + 1)} remaining frames have been added to the last cluster.")
                break

    return clusters

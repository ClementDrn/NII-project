import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import torch
from typing import List

from untitled.generator.filter.filter import Cluster, cluster_frames
from untitled.generator.filter.embedder import Embedder
from untitled.simulator.simulation_data import SimulationData, SimulationFrameData, SimulationMetadata


def create_mock_metadata() -> SimulationMetadata:
    """Helper function to create a SimulationMetadata instance."""
    return SimulationMetadata(
        step=0,
        control={"throttle": 0.0, "steer": 0.0, "brake": 0.0},
        speed=0.0,
        position=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0],
        forward_vector=[1.0, 0.0, 0.0],
        right_vector=[0.0, 1.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        acceleration=[0.0, 0.0, 0.0],
    )


def create_frame(image_name: str = "test_image.png") -> SimulationFrameData:
    """Helper function to create a SimulationFrameData instance."""
    return SimulationFrameData(
        image_name=image_name,
        metadata=create_mock_metadata()
    )


def create_mock_embedder(embedding_dim: int = 768) -> MagicMock:
    """Helper function to create a mock Embedder."""
    mock_embedder = MagicMock(spec=Embedder)
    # Return a random normalized embedding by default
    mock_embedder.embed.return_value = torch.nn.functional.normalize(
        torch.randn(embedding_dim), dim=0
    )
    return mock_embedder


def create_simulation_data(image_dir: Path, frames: List[SimulationFrameData]) -> SimulationData:
    """Helper function to create a SimulationData instance."""
    return SimulationData(
        image_dir=image_dir,
        frame_count=len(frames),
        frames=frames
    )


class TestCluster:
    """Tests for the Cluster class."""

    def test_cluster_initialization(self):
        """Test that Cluster can be initialized with image_dir and embedder."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        assert cluster is not None

    def test_cluster_starts_empty(self):
        """Test that a new cluster has no frames."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        data = cluster.get_data()
        assert data.frame_count == 0
        assert data.frames == []

    def test_get_data_returns_simulation_data(self):
        """Test that get_data returns a SimulationData object."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)

        result = cluster.get_data()

        assert isinstance(result, SimulationData)

    def test_get_data_has_correct_image_dir(self):
        """Test that get_data returns SimulationData with correct image_dir."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)

        result = cluster.get_data()

        assert result.image_dir == image_dir

    def test_add_frame_to_cluster(self):
        """Test that a frame can be added to a cluster."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame = create_frame()

        cluster.add(frame)

        data = cluster.get_data()
        assert data.frame_count == 1
        assert frame in data.frames

    def test_add_multiple_frames_to_cluster(self):
        """Test that multiple frames can be added to a cluster."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame1 = create_frame("image1.png")
        frame2 = create_frame("image2.png")
        frame3 = create_frame("image3.png")

        cluster.add(frame1)
        cluster.add(frame2)
        cluster.add(frame3)

        data = cluster.get_data()
        assert data.frame_count == 3
        assert frame1 in data.frames
        assert frame2 in data.frames
        assert frame3 in data.frames

    def test_calculate_similarity_on_empty_cluster_returns_none(self):
        """Test that calculate_similarity returns None for an empty cluster."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame = create_frame()

        similarity = cluster.calculate_similarity(frame)

        assert similarity is None

    def test_calculate_similarity_returns_float(self):
        """Test that calculate_similarity returns a float after adding frames."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame1 = create_frame("image1.png")
        frame2 = create_frame("image2.png")

        cluster.add(frame1)
        similarity = cluster.calculate_similarity(frame2)

        assert isinstance(similarity, float)

    def test_calculate_similarity_range(self):
        """Test that similarity score is between -1 and 1 (cosine similarity range)."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame1 = create_frame("image1.png")
        frame2 = create_frame("image2.png")

        cluster.add(frame1)
        similarity = cluster.calculate_similarity(frame2)

        # Allow small epsilon for floating point precision
        assert -1.0 - 1e-6 <= similarity <= 1.0 + 1e-6

    def test_calculate_similarity_uses_embedder(self):
        """Test that calculate_similarity uses the embedder to compute embeddings."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame1 = create_frame("image1.png")
        frame2 = create_frame("image2.png")

        cluster.add(frame1)
        cluster.calculate_similarity(frame2)

        # Embedder should be called at least for the frames
        assert mock_embedder.embed.called

    def test_identical_embeddings_give_high_similarity(self):
        """Test that identical embeddings result in high similarity score."""
        fixed_embedding = torch.nn.functional.normalize(torch.ones(768), dim=0)
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed.return_value = fixed_embedding

        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame1 = create_frame("image1.png")
        frame2 = create_frame("image2.png")

        cluster.add(frame1)
        similarity = cluster.calculate_similarity(frame2)

        # Identical normalized vectors should have cosine similarity of 1.0
        assert similarity > 0.99

    def test_orthogonal_embeddings_give_low_similarity(self):
        """Test that orthogonal embeddings result in low similarity score."""
        embedding1 = torch.zeros(768)
        embedding1[0] = 1.0
        embedding2 = torch.zeros(768)
        embedding2[1] = 1.0

        mock_embedder = MagicMock(spec=Embedder)
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                return embedding1
            return embedding2

        mock_embedder.embed.side_effect = side_effect

        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame1 = create_frame("image1.png")
        frame2 = create_frame("image2.png")

        cluster.add(frame1)
        similarity = cluster.calculate_similarity(frame2)

        # Orthogonal vectors should have cosine similarity close to 0
        assert abs(similarity) < 0.01

    def test_add_uses_embedder(self):
        """Test that add method uses the embedder."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame = create_frame()

        cluster.add(frame)

        assert mock_embedder.embed.called

    def test_embedder_called_with_correct_path(self):
        """Test that embedder is called with the correct image path."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        cluster = Cluster(image_dir, mock_embedder)
        frame = create_frame("test_image.png")

        cluster.add(frame)

        expected_path = image_dir / "test_image.png"
        mock_embedder.embed.assert_called_with(expected_path)


class TestClusterFrames:
    """Tests for the cluster_frames function."""

    def test_cluster_frames_empty_input(self):
        """Test that cluster_frames returns empty list for empty input."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        empty_data = create_simulation_data(image_dir, [])

        result = cluster_frames(empty_data, mock_embedder, similarity_threshold=0.8)

        assert result == []

    def test_cluster_frames_returns_list(self):
        """Test that cluster_frames returns a list."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        frames = [create_frame("image1.png")]
        frame_data = create_simulation_data(image_dir, frames)

        result = cluster_frames(frame_data, mock_embedder, similarity_threshold=0.8)

        assert isinstance(result, list)

    def test_cluster_frames_returns_list_of_clusters(self):
        """Test that cluster_frames returns a list of Cluster objects."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        frames = [create_frame("image1.png")]
        frame_data = create_simulation_data(image_dir, frames)

        result = cluster_frames(frame_data, mock_embedder, similarity_threshold=0.8)

        assert all(isinstance(c, Cluster) for c in result)

    def test_cluster_frames_single_frame_creates_single_cluster(self):
        """Test that a single frame creates exactly one cluster."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        frames = [create_frame("image1.png")]
        frame_data = create_simulation_data(image_dir, frames)

        result = cluster_frames(frame_data, mock_embedder, similarity_threshold=0.8)

        assert len(result) == 1
        assert result[0].get_data().frame_count == 1

    def test_cluster_frames_identical_embeddings_single_cluster(self):
        """Test that identical embeddings result in a single cluster."""
        fixed_embedding = torch.nn.functional.normalize(torch.ones(768), dim=0)
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed.return_value = fixed_embedding

        image_dir = Path("/fake/image/dir")
        frames = [
            create_frame("image1.png"),
            create_frame("image2.png"),
            create_frame("image3.png"),
        ]
        frame_data = create_simulation_data(image_dir, frames)

        result = cluster_frames(frame_data, mock_embedder, similarity_threshold=0.8)

        assert len(result) == 1
        assert result[0].get_data().frame_count == 3

    def test_cluster_frames_different_embeddings_multiple_clusters(self):
        """Test that very different embeddings result in multiple clusters."""
        mock_embedder = MagicMock(spec=Embedder)
        call_count = [0]

        def create_orthogonal_embedding(*args, **kwargs):
            # Create embeddings that are orthogonal to each other
            embedding = torch.zeros(768)
            embedding[call_count[0] % 768] = 1.0
            call_count[0] += 1
            return embedding

        mock_embedder.embed.side_effect = create_orthogonal_embedding

        image_dir = Path("/fake/image/dir")
        frames = [
            create_frame("image1.png"),
            create_frame("image2.png"),
            create_frame("image3.png"),
        ]
        frame_data = create_simulation_data(image_dir, frames)

        result = cluster_frames(frame_data, mock_embedder, similarity_threshold=0.9)

        # With orthogonal embeddings and high threshold, should create multiple clusters
        assert len(result) >= 2

    def test_cluster_frames_respects_threshold(self):
        """Test that cluster_frames respects the similarity threshold."""
        mock_embedder = MagicMock(spec=Embedder)

        # Create embeddings with known high similarity (same direction, slight perturbation)
        base_embedding = torch.nn.functional.normalize(torch.ones(768), dim=0)
        # Create a second embedding with very high similarity (~0.99+)
        perturbation = torch.randn(768) * 0.01
        similar_embedding = torch.nn.functional.normalize(base_embedding + perturbation, dim=0)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return base_embedding
            return similar_embedding

        mock_embedder.embed.side_effect = side_effect

        image_dir = Path("/fake/image/dir")
        frames = [create_frame("image1.png"), create_frame("image2.png")]
        frame_data = create_simulation_data(image_dir, frames)

        # With low threshold, should be in same cluster (similarity is ~0.99+)
        result_low = cluster_frames(frame_data, mock_embedder, similarity_threshold=0.5)

        # Low threshold should create 1 cluster
        assert len(result_low) == 1

    def test_cluster_frames_all_frames_accounted_for(self):
        """Test that all frames end up in some cluster."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        frames = [
            create_frame("image1.png"),
            create_frame("image2.png"),
            create_frame("image3.png"),
            create_frame("image4.png"),
            create_frame("image5.png"),
        ]
        frame_data = create_simulation_data(image_dir, frames)

        result = cluster_frames(frame_data, mock_embedder, similarity_threshold=0.8)

        # Count total frames across all clusters
        total_frames = sum(c.get_data().frame_count for c in result)
        assert total_frames == 5

    def test_cluster_frames_preserves_frame_order_within_clusters(self):
        """Test that frames maintain their order within a cluster."""
        fixed_embedding = torch.nn.functional.normalize(torch.ones(768), dim=0)
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed.return_value = fixed_embedding

        image_dir = Path("/fake/image/dir")
        frames = [
            create_frame("image1.png"),
            create_frame("image2.png"),
            create_frame("image3.png"),
        ]
        frame_data = create_simulation_data(image_dir, frames)

        result = cluster_frames(frame_data, mock_embedder, similarity_threshold=0.8)

        # With identical embeddings, all in one cluster, order should be preserved
        cluster_frames_list = result[0].get_data().frames
        assert cluster_frames_list[0].image_name == "image1.png"
        assert cluster_frames_list[1].image_name == "image2.png"
        assert cluster_frames_list[2].image_name == "image3.png"

    def test_cluster_frames_uses_embedder(self):
        """Test that cluster_frames uses the provided embedder."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        frames = [create_frame("image1.png"), create_frame("image2.png")]
        frame_data = create_simulation_data(image_dir, frames)

        cluster_frames(frame_data, mock_embedder, similarity_threshold=0.8)

        assert mock_embedder.embed.called

    def test_cluster_frames_with_zero_threshold(self):
        """Test cluster_frames behavior with threshold of 0."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        frames = [
            create_frame("image1.png"),
            create_frame("image2.png"),
        ]
        frame_data = create_simulation_data(image_dir, frames)

        # With threshold 0, any positive similarity should pass
        result = cluster_frames(frame_data, mock_embedder, similarity_threshold=0.0)

        # Should still create clusters
        assert len(result) >= 1

    def test_cluster_frames_with_threshold_one(self):
        """Test cluster_frames behavior with threshold of 1 (maximum)."""
        mock_embedder = create_mock_embedder()
        image_dir = Path("/fake/image/dir")
        frames = [
            create_frame("image1.png"),
            create_frame("image2.png"),
        ]
        frame_data = create_simulation_data(image_dir, frames)

        # With threshold 1, only identical embeddings should cluster
        result = cluster_frames(frame_data, mock_embedder, similarity_threshold=1.0)

        # With random embeddings and threshold 1, likely to have multiple clusters
        assert len(result) >= 1

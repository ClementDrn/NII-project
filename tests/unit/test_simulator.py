import pytest
from tests.fakes import FakeSimulator


class TestSimulator:
    """
    Test the Simulator class using FakeSimulator.
    
    Uses FakeSimulator to avoid CARLA dependencies while testing
    path calculation logic that is identical in both implementations.
    """

    @pytest.fixture
    def simulator(self):
        return FakeSimulator()

    # --- Test the calculate_scenario_base_dir and calculate_run_base_dir methods ---
    def test_calculate_run_base_dir_no_subdir(self, simulator):
        """
        Test the calculate_run_base_dir method.
        """
        # base_dir: Path, job_id: int, scenario_id: int, run_id: int, has_job_subdir: bool = True
        base_dir = "/path/to/base/dir"
        job_id = 123
        scenario_id = 456
        run_id = 789
        has_job_subdir = False
        expected_dir = "/path/to/base/dir/job_123/sce_456/run_789"
        assert simulator._calculate_run_base_dir(base_dir, job_id, scenario_id, run_id, has_job_subdir) == expected_dir

    def test_calculate_run_base_dir(self, simulator):
        """
        Test the calculate_run_base_dir method.
        """
        # base_dir: Path, job_id: int, scenario_id: int, run_id: int, has_job_subdir: bool = True
        base_dir = "/path/to/base/dir"
        job_id = 123
        scenario_id = 456
        run_id = 789
        has_job_subdir = True
        expected_dir = "/path/to/base/dir/job_123/sce_456/run_789"
        assert simulator._calculate_run_base_dir(base_dir, job_id, scenario_id, run_id, has_job_subdir) == expected_dir

    def test_calculate_run_base_dir_plus_subdir(self, simulator):
        """
        Test the calculate_run_base_dir method.
        """
        # base_dir: Path, job_id: int, scenario_id: int, run_id: int, has_job_subdir: bool = True
        base_dir = "/path/to/base/dir"
        job_id = 123
        scenario_id = 456
        run_id = 789
        has_job_subdir = True
        simulator.set_base_job_subdir("job_subdir")
        expected_dir = "/path/to/base/dir/job_123/job_subdir/sce_456/run_789"
        assert simulator._calculate_run_base_dir(base_dir, job_id, scenario_id, run_id, has_job_subdir) == expected_dir
        
    # --- Test directory hierarchy creation -----------------------------------

    def test_directory_hierarchy_creation_1_scenario_and_groundtruth(self, fake_simulator_with_fs, temp_base_dir):
        """
        Test that the directory hierarchy is created correctly.
        """
        simulator = fake_simulator_with_fs
        
        # Simulate groundtruths
        simulator.set_base_job_subdir("gt")
        simulator.set_job_context(123, 1)
        simulator.set_scenario(1)
        simulator.run_scenario()
        simulator.run_scenario()
        
        # Simulate experiments (no subdir)
        simulator.set_base_job_subdir("")
        simulator.set_job_context(123, 1)
        simulator.set_scenario(1)
        simulator.run_scenario()
        simulator.run_scenario()

        # Check that the expected directories were created in the output directory
        output_dir = temp_base_dir / "output"
        expected_paths = [
            output_dir / "job_123" / "gt" / "sce_001" / "run_000",
            output_dir / "job_123" / "gt" / "sce_001" / "run_001",
            output_dir / "job_123" / "sce_001" / "run_000",
            output_dir / "job_123" / "sce_001" / "run_001",
        ]
        for expected_path in expected_paths:
            assert expected_path.exists() and expected_path.is_dir(), f"Expected directory {expected_path} to be created"

    def test_directory_hierarchy_creation_2_scenarios_and_groundtruth(self, fake_simulator_with_fs, temp_base_dir):
        """
        Test that the directory hierarchy is created correctly.
        """
        simulator = fake_simulator_with_fs
        
        # Simulate groundtruths
        simulator.set_base_job_subdir("gt")
        # Scenario 1 with 2 runs
        simulator.set_job_context(123, 1)
        simulator.set_scenario(1)
        simulator.run_scenario()
        simulator.run_scenario()
        # Scenario 2 with 1 run
        simulator.set_job_context(123, 2)
        simulator.set_scenario(2)
        simulator.run_scenario()

        # Simulate experiments (no subdir)
        simulator.set_base_job_subdir("")
        # Scenario 1 with 2 runs
        simulator.set_job_context(123, 1)
        simulator.set_scenario(1)
        simulator.run_scenario()
        simulator.run_scenario()
        # Scenario 2 with 1 run
        simulator.set_job_context(123, 2)
        simulator.set_scenario(2)
        simulator.run_scenario()    

        # Check that the expected directories were created in the output directory
        output_dir = temp_base_dir / "output"
        expected_paths = [
            output_dir / "job_123" / "gt" / "sce_001" / "run_000",
            output_dir / "job_123" / "gt" / "sce_001" / "run_001",
            output_dir / "job_123" / "gt" / "sce_002" / "run_000",
            output_dir / "job_123" / "sce_001" / "run_000",
            output_dir / "job_123" / "sce_001" / "run_001",
            output_dir / "job_123" / "sce_002" / "run_000",
        ]
        for expected_path in expected_paths:
            assert expected_path.exists() and expected_path.is_dir(), f"Expected directory {expected_path} to be created"
"""Validate shipped configuration without reading private .env or hosts files."""

import logging
import shutil
from pathlib import Path

import pytest
import yaml

from digit.core.job_handler import JobHandler
from digit.core.settings import HostConfig, configure_settings_for_job, load_config
from digit.core.version import (
    CONFIG_VERSION,
    HOSTS_VERSION,
    JOBS_VERSION,
    are_versions_compatible,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    shutil.copy(PROJECT_ROOT / "config.yaml", tmp_path / "config.yaml")
    # No .env is copied, and no host credentials are required for schema validation.
    monkeypatch.delenv("HOSTS_PATH", raising=False)
    return load_config(tmp_path, logging.getLogger(__name__))


@pytest.mark.parametrize(
    ("filename", "version"),
    [
        ("config.yaml", CONFIG_VERSION),
        ("hosts.example.yaml", HOSTS_VERSION),
        ("jobs.yaml", JOBS_VERSION),
    ],
)
def test_shipped_yaml_versions(filename, version):
    data = yaml.safe_load((PROJECT_ROOT / filename).read_text())
    assert are_versions_compatible(data["version"], version)


def test_example_hosts_validate():
    data = yaml.safe_load((PROJECT_ROOT / "hosts.example.yaml").read_text())
    hosts = [HostConfig(**host) for host in data["hosts"]]
    assert hosts
    assert len({host.id for host in hosts}) == len(hosts)


def test_example_environment_documents_required_settings():
    values = dict(
        line.split("=", 1)
        for line in (PROJECT_ROOT / ".env.example").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert {
        "CARLA_PATH",
        "CONDA_PATH",
        "OUTPUT_PATH",
        "CARLA_PORT",
        "CARLA_TRAFFIC_MANAGER_PORT",
        "HOSTS_PATH",
    } <= values.keys()
    assert all(values.values())


def test_config_ids_are_unique_and_models_reference_existing_steps(settings):
    assert settings.hosts == []
    for configs in (
        settings.filters,
        settings.generators,
        settings.simulators,
        settings.model_instances,
    ):
        assert configs
        assert len({config.id for config in configs}) == len(configs)
    step_ids = {
        step.id for generator in settings.generators for step in generator.steps
    }
    for instance in settings.model_instances:
        assert bool(instance.local_config) != bool(instance.remote_config)
        assert set(instance.step_ids) <= step_ids
    assert {generator.design for generator in settings.generators} <= {
        "rule_based",
        "single_step_vlm",
        "multi_step_vlm",
    }


@pytest.mark.parametrize(
    "jobs_file", ["jobs.yaml", "baseline.yaml"]
)
def test_shipped_jobs_resolve_configs_and_active_models(settings, jobs_file):
    jobs = JobHandler(str(PROJECT_ROOT / jobs_file)).jobs
    assert jobs
    for job in jobs:
        configure_settings_for_job(settings, job)
        assert settings.simulator.id == job.simulator
        assert [
            instance.id for instance in settings.active_model_instances
        ] == job.model_instances
        if job.filter:
            assert settings.filter.id == job.filter
        if job.generator:
            assert settings.generator.id == job.generator
            if settings.generator.design != "rule_based":
                for step in settings.generator.steps:
                    assert any(
                        step.id in instance.step_ids
                        for instance in settings.active_model_instances
                    )


def test_example_hosts_cover_every_remote_model(settings):
    data = yaml.safe_load((PROJECT_ROOT / "hosts.example.yaml").read_text())
    hosts = {host["id"] for host in data["hosts"]}
    referenced = {
        instance.remote_config.host_id
        for instance in settings.model_instances
        if instance.remote_config is not None
    }
    assert referenced <= hosts

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import asyncio
import os
import shutil
import socket
import tempfile

import yaml

from untitled.core.base import CONFIG, LOG
from untitled.core.settings import HostConfig, ModelInstanceConfig, Settings
from untitled.generator.model_config import get_model_config


@dataclass
class ProxyModelEntry:
    """LiteLLM proxy entry for a single model candidate."""

    model_name: str
    api_base: str
    api_key: str
    extra_params: dict[str, Any] = field(default_factory=dict)


class ModelProxyManager:
    """Manage a LiteLLM proxy subprocess for the current job."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4000):
        self._host = host
        self._port = port
        self._process: asyncio.subprocess.Process | None = None
        self._config_file: Path | None = None
        self._work_dir: Path | None = None
        self._lock = asyncio.Lock()

    def get_proxy_api_base(self) -> str:
        """Return the OpenAI-compatible base URL exposed by the proxy."""
        return f"http://{self._host}:{self._port}/v1"

    def _resolve_upstream_api_base(self, model_name: str, host: str) -> str:
        """Resolve the upstream API base for a model candidate.

        The proxy owns provider-specific URL handling now:
        - If the host already includes a path, keep it.
        - OpenAI-compatible providers use `/v1`.
        - Gemini and Anthropic use the host root because LiteLLM handles them natively.
        - Any other provider falls back to the host root.
        """
        host_base = host.rstrip("/")

        parsed_host = urlparse(host_base)
        host_has_explicit_path = parsed_host.path not in ("", "/")
        if host_has_explicit_path:
            return host_base

        provider = ""
        if model_name:
            try:
                from litellm import get_llm_provider

                _, provider, _, _ = get_llm_provider(
                    model=model_name,
                    api_base=host_base,
                    api_key="EMPTY",
                )
            except Exception:
                provider = model_name.split("/", 1)[0].lower() if "/" in model_name else ""

        openai_compatible_providers = {"", "openai", "text-completion-openai", "azure", "azure_ai"}
        if provider in openai_compatible_providers:
            return f"{host_base}/v1"

        provider_default_base = {"gemini", "anthropic"}
        if provider in provider_default_base:
            return host_base

        return host_base

    def _resolve_host_base(self, host: str) -> str:
        """Normalize host URLs so the proxy config stays consistent."""
        return host.rstrip("/")

    def _build_model_entries(self, settings: Settings) -> list[ProxyModelEntry]:
        """Create one proxy entry per active model instance."""
        host_lookup: dict[str, HostConfig] = {host.id: host for host in settings.hosts}
        entries: list[ProxyModelEntry] = []

        for instance in settings.active_model_instances:
            model_config = get_model_config(instance.model_name)
            if instance.local_config is not None:
                # Local models are already OpenAI-compatible through their own server.
                api_base = f"http://127.0.0.1:{instance.local_config.port}/v1"
                api_key = "EMPTY"
            elif instance.remote_config is not None:
                host_config = host_lookup.get(instance.remote_config.host_id)
                if host_config is None:
                    raise ValueError(f"Host with ID '{instance.remote_config.host_id}' not found in settings.hosts.")
                api_base = self._resolve_upstream_api_base(model_config.model_name, host_config.host)
                api_key = host_config.api_key
            else:
                continue

            entries.append(
                ProxyModelEntry(
                    model_name=model_config.model_name,
                    api_base=api_base,
                    api_key=api_key,
                )
            )

        return entries

    def _build_config(self, settings: Settings) -> dict[str, Any]:
        """Build a temporary LiteLLM proxy configuration from the active job."""
        model_list = []
        for entry in self._build_model_entries(settings):
            model_list.append(
                {
                    "model_name": entry.model_name,
                    "litellm_params": {
                        "model": entry.model_name,
                        "api_base": entry.api_base,
                        "api_key": entry.api_key,
                        **entry.extra_params,
                    },
                }
            )

        return {
            "model_list": model_list,
            "litellm_settings": {
                "drop_params": True,
            },
        }

    async def _wait_for_proxy_ready(self) -> None:
        """Poll the proxy TCP port until it becomes reachable."""
        max_wait_seconds = 180
        poll_interval_seconds = 1.0
        elapsed_seconds = 0.0

        while elapsed_seconds < max_wait_seconds:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self._host, self._port),
                    timeout=5,
                )
                writer.close()
                await writer.wait_closed()
                return
            except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
                if self._process is not None and self._process.returncode is not None:
                    stdout, stderr = await self._process.communicate()
                    message = f"LiteLLM proxy exited early with code {self._process.returncode}"
                    if stdout:
                        message += f"\nStdout: {stdout.decode()}"
                    if stderr:
                        message += f"\nStderr: {stderr.decode()}"
                    raise RuntimeError(message)

                await asyncio.sleep(poll_interval_seconds)
                elapsed_seconds += poll_interval_seconds

        raise TimeoutError(f"LiteLLM proxy did not become ready within {max_wait_seconds} seconds.")

    async def start(self, settings: Settings | None = None) -> None:
        """Start the proxy once for the current job."""
        if settings is None:
            settings = CONFIG

        async with self._lock:
            if self._process is not None and self._process.returncode is None:
                return

            model_list = self._build_model_entries(settings)
            if not model_list:
                LOG.info("No active model candidates found; LiteLLM proxy will not be started.")
                return

            # Keep the proxy config isolated per job so it can be cleaned up safely.
            self._work_dir = Path(tempfile.mkdtemp(prefix="untitled_litellm_proxy_"))
            self._config_file = self._work_dir / "litellm_proxy_config.yaml"
            config_data = self._build_config(settings)
            self._config_file.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

            litellm_executable = shutil.which("litellm")
            if litellm_executable is None:
                raise RuntimeError("Could not find the 'litellm' executable in PATH.")

            command = [
                litellm_executable,
                "--config",
                str(self._config_file),
                "--host",
                self._host,
                "--port",
                str(self._port),
            ]

            LOG.info("Starting LiteLLM proxy with %d model candidates on %s:%d.", len(model_list), self._host, self._port)
            LOG.debug("LiteLLM proxy config file: %s", self._config_file)
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    "HOME": os.environ.get("HOME", ""),
                    "USER": os.environ.get("USER", ""),
                    "LOGNAME": os.environ.get("LOGNAME", ""),
                    "SHELL": os.environ.get("SHELL", "/bin/bash"),
                    "TERM": os.environ.get("TERM", "xterm"),
                    "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
                },
            )

            await self._wait_for_proxy_ready()
            LOG.info("LiteLLM proxy is ready on %s", self.get_proxy_api_base())

    async def stop(self) -> None:
        """Stop the proxy and remove its temporary config."""
        async with self._lock:
            if self._process is not None:
                if self._process.returncode is None:
                    self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        LOG.warning("LiteLLM proxy did not stop gracefully, killing it.")
                        self._process.kill()
                        await self._process.wait()
                self._process = None

            if self._work_dir is not None:
                shutil.rmtree(self._work_dir, ignore_errors=True)
                self._work_dir = None
                self._config_file = None


_MODEL_PROXY_MANAGER: ModelProxyManager | None = None


def get_model_proxy_manager() -> ModelProxyManager:
    """Return the shared proxy manager for the current process."""
    global _MODEL_PROXY_MANAGER
    if _MODEL_PROXY_MANAGER is None:
        _MODEL_PROXY_MANAGER = ModelProxyManager()
    return _MODEL_PROXY_MANAGER

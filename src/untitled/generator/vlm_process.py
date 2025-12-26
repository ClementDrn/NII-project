from untitled.core.base import CONFIG, LOG, VENDOR_VLM_DIR
from untitled.generator.model_config import ModelConfig
from untitled.core.settings import LocalVLMConfig

import asyncio
from enum import Enum
from urllib.parse import urlparse
import os
import json


class VLMProcess:
    """
    Represents a VLM process.
    """
    
    class State(Enum):
        STOPPED = 0
        STARTING = 1
        RUNNING = 2
        STOPPING = 3

    def __init__(self, host: str, model_config: ModelConfig, process_config: LocalVLMConfig):
        # State
        self._state: VLMProcess.State = VLMProcess.State.STOPPED
        self._lock: asyncio.Lock = asyncio.Lock()

        # Subprocess handle
        self._process: asyncio.subprocess.Process = None

        # vLLM parameters
        self._host = host
        self._model_config = model_config
        self._process_config = process_config

    async def _start_process(self):
        """
        Internal method to start the VLM process.
        """
                # Parse host URL to get host and port
        parsed = urlparse(self._host)
        host_addr = parsed.hostname or "0.0.0.0"
        port = parsed.port or 8000
        
        # Build vLLM command with activation
        cmd = [
            "bash", "-c",
            f"source {VENDOR_VLM_DIR}/.venv/bin/activate "
            # Model config params
            f"&& vllm serve {self._model_config.model_name} "
            f"--served-model-name {self._model_config.served_name} "
            f"--max-model-len {self._model_config.max_model_len} "
            # Process config params (specific to this instance)
            f"--tensor-parallel-size {self._process_config.tensor_parallel_size} "
            f"--gpu-memory-utilization {self._process_config.gpu_memory_utilization} "
            f"--max-num-seqs {self._process_config.max_num_seqs} "
            f"--dtype {self._process_config.dtype} "
            # Interface params
            f"--host {host_addr} "
            f"--port {port} "
            f"--trust-remote-code "
        ]
        # Add limit-mm-per-prompt
        limit_mm = json.dumps(self._model_config.limit_mm_per_prompt)
        cmd[2] += f" --limit-mm-per-prompt '{limit_mm}'"
        # Add optional flags to the command string
        if self._process_config.enable_chunked_prefill:
            cmd[2] += " --enable-chunked-prefill"
        if self._process_config.enable_prefix_caching:
            cmd[2] += " --enable-prefix-caching"
        
        LOG.debug(f"vLLM command: {' '.join(cmd)}")
        
        # Start process with clean environment to avoid conda/venv interference
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                'HOME': os.environ.get('HOME', ''),
                'USER': os.environ.get('USER', ''),
                'LOGNAME': os.environ.get('LOGNAME', ''),
                'SHELL': os.environ.get('SHELL', '/bin/bash'),
                'TERM': os.environ.get('TERM', 'xterm'),
                'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            }
        )
        
        # Wait for server to be ready by checking TCP connection
        max_wait_time = 180  # 3 minutes timeout
        check_interval = 2  # Check every 2 seconds
        elapsed_time = 0
        
        LOG.info(f"Waiting for VLM server to be ready on {host_addr}:{port}...")
        while elapsed_time < max_wait_time:
            try:
                # Try to establish a TCP connection
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host_addr, port),
                    timeout=5
                )
                writer.close()
                await writer.wait_closed()
                LOG.info(f"VLM server is ready on {host_addr}:{port}")
                break
            except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
                # Server not ready yet, wait and retry
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval
                
                # Check if process has died
                if self._process.returncode is not None:
                    # Process has terminated unexpectedly
                    stdout, stderr = await self._process.communicate()
                    error_msg = f"VLM process terminated unexpectedly with code {self._process.returncode}"
                    if stderr:
                        error_msg += f"\nStderr: {stderr.decode()}"
                    LOG.error(error_msg)
                    raise RuntimeError(error_msg)
        else:
            # Timeout reached
            LOG.error(f"Timeout waiting for VLM server to start after {max_wait_time}s")
            # Kill the process if it's still running
            if self._process.returncode is None:
                self._process.kill()
                await self._process.wait()
            raise TimeoutError(f"VLM server did not start within {max_wait_time} seconds")
        

    async def _stop_process(self):
        """
        Internal method to stop the VLM process.
        """
        # Terminate the process
        if self._process:
            LOG.debug("Stopping VLM process...")
            try:
                self._process.terminate()

                # Wait for process to exit
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    LOG.warning("VLM process did not stop gracefully, killing process...")
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                LOG.debug("Process already terminated")

            LOG.debug("VLM process stopped")
            self._process = None


    async def start(self):
        """
        Start the VLM process.
        Only one start operation will be processed at a time.
        If already running or starting, this will wait and return without action.
        """
        async with self._lock:
            # Check that current state is not a transition state (should not happen)
            if self._state in (VLMProcess.State.STARTING, VLMProcess.State.STOPPING):
                LOG.warning("VLM process is in a transition state while starting. This should not happen.")
                return

            # If already running, do nothing
            if self._state == VLMProcess.State.RUNNING:
                return
            
            try:
                # Transition to STARTING and start process
                self._state = VLMProcess.State.STARTING
                await self._start_process()
                
                # Transition to RUNNING
                self._state = VLMProcess.State.RUNNING
            except Exception as e:
                # On error, revert to STOPPED
                self._state = VLMProcess.State.STOPPED
                LOG.error("Failed to start VLM process", exc_info=e)

    async def stop(self):
        """
        Stop the VLM process.
        Only one stop operation will be processed at a time.
        If already stopped or stopping, this will wait and return without action.
        """
        async with self._lock:
            # Check that current state is not a transition state (should not happen)
            if self._state in (VLMProcess.State.STARTING, VLMProcess.State.STOPPING):
                LOG.warning("VLM process is in a transition state while stopping. This should not happen.")
                return

            # If already stopped, do nothing
            if self._state == VLMProcess.State.STOPPED:
                return
            
            try:
                # Transition to STOPPING and stop process
                self._state = VLMProcess.State.STOPPING
                await self._stop_process()
                
                # Transition to STOPPED
                self._state = VLMProcess.State.STOPPED
            except Exception as e:
                # On error, still mark as STOPPED
                self._state = VLMProcess.State.STOPPED
                LOG.error("Failed to stop VLM process", exc_info=e)

    def get_state(self) -> State:
        """
        Get the current state of the VLM process.
        """
        return self._state

    def is_parallel_to_simulator_supported(self) -> bool:
        """
        Returns True if the VLM process supports parallel operation with the simulator.
        """
        return self._process_config.is_parallel_to_simulator
    
    def get_model_name(self) -> str:
        """
        Get the model name served by this VLM process.
        """
        return self._model_config.model_name

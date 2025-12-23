from untitled.generator.vlm_client import VLMClient
from untitled.core.base import LOG

from typing import List
import asyncio
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PromptRequest:
    """A queued prompt request."""
    system_text: str
    user_text: str
    user_image_paths: list[Path]
    future: asyncio.Future


class PromptDispatcher:
    """
    Dispatch prompts to multiple VLM clients.
    """

    @dataclass
    class ClientResource:
        """
        Internal class to track VLM client resource usage.
        Sortable by client priority for use in PriorityQueue.
        """
        client: VLMClient
        is_idle: asyncio.Event
        is_removal_requested: bool = False
        
        def __lt__(self, other: 'PromptDispatcher.ClientResource') -> bool:
            """Compare by client priority for PriorityQueue ordering."""
            return self.client.priority < other.client.priority

    def __init__(self, vlm_clients: List[VLMClient] = []):
        # List of all VLM clients to use for processing prompts
        if vlm_clients is None:
            vlm_clients = []
        self._all_clients: List[PromptDispatcher.ClientResource] = [
            self.ClientResource(client=client, is_idle=asyncio.Event()) 
            for client in vlm_clients
        ]
        # Set all clients as initially idle
        for resource in self._all_clients:
            resource.is_idle.set()
        self._free_clients: asyncio.PriorityQueue[PromptDispatcher.ClientResource] = asyncio.PriorityQueue()    # sorted by client priority
        for resource in self._all_clients:
            self._free_clients.put_nowait(resource)

        # Queue of prompts to be processed
        self._prompt_queue: asyncio.Queue[PromptRequest] = asyncio.Queue()

        # Dispatcher task
        self._dispatcher_task: asyncio.Task = None


    async def _send_prompt_to_vlm(self, prompt: PromptRequest, client_resource: ClientResource):
        """
        Send a prompt to one of the VLM clients in a round-robin fashion.
        """
        try:
            # Send prompt
            response = await client_resource.client.prompt(prompt.system_text, prompt.user_text, prompt.user_image_paths)
            # Set the result in the future
            prompt.future.set_result(response)
        except Exception as e:
            prompt.future.set_exception(e)
        finally:
            # Mark prompt as done (to wake up join() waiters)
            self._prompt_queue.task_done()

    async def _dispatch_prompts_to_client(self, prompts: List[PromptRequest], client_resource: ClientResource):
        """
        Dispatch a list of prompts to a specific VLM client.
        """
        # Dispatch all prompts to the VLM client concurrently
        tasks = [self._send_prompt_to_vlm(prompt, client_resource) for prompt in prompts]
        await asyncio.gather(*tasks)

        # Mark client as idle and available again, unless removal was requested
        if not client_resource.is_removal_requested:
            self._free_clients.put_nowait(client_resource)
        client_resource.is_idle.set()

    async def _dispatcher_loop(self):
        """
        Main dispatcher loop to continuously process prompts.
        """
        try:
            while True:
                # 1. Wait for a prompt to be available
                LOG.debug("Prompt dispatcher waiting for a prompt...")
                prompt = await self._prompt_queue.get()
                LOG.debug("Prompt dispatcher fetched a prompt from the queue.")

                # 2. Wait for an available client resource
                # The fetched client has the highest priority (lowest priority value)
                client_resource: PromptDispatcher.ClientResource = None
                while client_resource is None:
                    client_resource = await self._free_clients.get()
                    LOG.debug(f"Prompt dispatcher found client.")
                    # If this client is being removed, skip it
                    if client_resource.is_removal_requested:
                        client_resource = None
                
                # Mark client as busy
                LOG.debug(f"Prompt dispatcher clearing idle status for client...")
                client_resource.is_idle.clear()

                # 3. Fetch as many prompts as the client can handle
                LOG.debug(f"Prompt dispatcher preparing prompts for client...")
                prompts = [prompt]
                prompt_max_count = client_resource.client.batch_size
                for _ in range(prompt_max_count - 1):
                    try:
                        fetched_prompt: PromptRequest = self._prompt_queue.get_nowait()
                        prompts.append(fetched_prompt)
                    except asyncio.QueueEmpty:   # no more prompts available
                        break
                
                # 4. Dispatch prompts to the VLM client (don't wait)
                LOG.debug(f"Prompt dispatcher dispatching {len(prompts)} prompts to client...")
                asyncio.create_task(self._dispatch_prompts_to_client(prompts, client_resource))
                LOG.debug(f"Prompt dispatcher finished one loop iteration.")
        except asyncio.CancelledError:
            LOG.debug("Prompt dispatcher loop cancelled.")


    async def start(self):
        """
        Start the prompt dispatcher loop.
        """
        if self._dispatcher_task is not None:
            LOG.warning("Prompt dispatcher start called but dispatcher is already running.")
            return
        
        # Start the dispatcher loop
        self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())

    async def stop(self):
        """
        Stop the prompt dispatcher.
        """
        # If dispatcher was never started or already stopped, do nothing
        if self._dispatcher_task is None:
            LOG.warning("Prompt dispatcher stop called but dispatcher is not running.")
            return
            
        # Request stop
        self._dispatcher_task.cancel()
        
        # Wait until the dispatcher loop has stopped
        try:
            await self._dispatcher_task
        except asyncio.CancelledError:
            pass
        
        # Mark task as stopped
        self._dispatcher_task = None
        
        # Wait until all tasks are done
        await asyncio.gather(*[
            resource.is_idle.wait() for resource in self._all_clients
        ])

    async def is_running(self) -> bool:
        """
        Check if the prompt dispatcher is running.
        """
        return self._dispatcher_task is not None and not self._dispatcher_task.done()

    async def add(self, prompt: PromptRequest):
        """
        Add a prompt to the buffer.
        """
        await self._prompt_queue.put(prompt)

    async def add_clients(self, new_clients: List[VLMClient]):
        """
        Add VLM clients to the list of available clients.
        """
        # Create new resources and add them to the main list
        new_resources = [
            self.ClientResource(client=client, is_idle=asyncio.Event()) 
            for client in new_clients
        ]
        # Set all new clients as initially idle
        for resource in new_resources:
            resource.is_idle.set()
        
        self._all_clients.extend(new_resources)

        # Add new resources to the free clients queue
        for resource in new_resources:
            self._free_clients.put_nowait(resource)

    async def remove_clients(self, old_clients: List[VLMClient]):
        """
        Remove VLM clients from the list of available clients.

        The function returns when the clients are effectively removed.
        The function may take time to return because it has to wait for each of the concerned clients to finish their task first.
        """
        # Set removal flag for each client to be removed
        resources_to_remove = []
        resources_to_keep = []
        for resource in self._all_clients:
            if resource.client in old_clients:
                resources_to_remove.append(resource)
                resource.is_removal_requested = True
            else:
                resources_to_keep.append(resource)

        # Wait until all removed clients are no longer processing prompts
        await asyncio.gather(*[
            resource.is_idle.wait() for resource in resources_to_remove
        ])

        # Remove clients from the main list
        self._all_clients = resources_to_keep

    async def get_clients(self) -> List[VLMClient]:
        """
        Get the list of currently registered VLM clients.
        """
        return [resource.client for resource in self._all_clients]

    async def wait_until_all_prompts_processed(self):
        """
        Wait until all prompts in the queue have been processed.
        """
        await self._prompt_queue.join()

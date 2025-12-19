from unittest.mock import patch, MagicMock

from untitled.generator.vlm import create_vlm_client
from untitled.core.settings import ModelInstanceConfig, LocalVLMConfig, RemoteVLMConfig


def test_create_vlm_client_returns_local_client():
    with patch("untitled.generator.vlm.get_model_config") as mock_get_config:
        mock_get_config.return_value = MagicMock()
        with patch("untitled.generator.vlm.LocalVLM") as mock_local:
            # Create local instance config
            local_config = LocalVLMConfig(port=8000)
            instance_config = ModelInstanceConfig(
                id="test-local",
                model_name="test-model",
                type="vlm",
                local_config=local_config
            )
            create_vlm_client(instance_config)
            mock_local.assert_called_once()


def test_create_vlm_client_returns_remote_client():
    with patch("untitled.generator.vlm.get_model_config") as mock_get_config:
        mock_get_config.return_value = MagicMock()
        with patch("untitled.generator.vlm.RemoteVLM") as mock_remote:
            with patch("untitled.generator.vlm.get_host_config_by_id") as mock_get_host:
                mock_get_host.return_value = MagicMock(host="http://remote-host", api_key="token")
                # Create remote instance config
                remote_config = RemoteVLMConfig(host_id="test-host")
                instance_config = ModelInstanceConfig(
                    id="test-remote",
                    model_name="test-model",
                    type="vlm",
                    remote_config=remote_config
                )
                create_vlm_client(instance_config)
                mock_remote.assert_called_once()

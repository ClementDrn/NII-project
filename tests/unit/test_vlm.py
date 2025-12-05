import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock, call
from pathlib import Path
import base64
import io

# Mock the CONFIG before importing any modules that depend on it
with patch('untitled.core.settings.load_config') as mock_load_config:
    mock_config = MagicMock()
    mock_config.vlm.host = "http://localhost:8000"
    mock_config.vlm.api_key = "test-key"
    mock_load_config.return_value = mock_config
    
    from untitled.generator.vlm import VLM, encode_image_to_base64, parse_vlm_response


# --- Fixtures and Mocks ------------------------------------------------------

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client with chat completions."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "<think>Reasoning...</think>\n\nFinal answer."
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_image_file():
    """Mock image file for base64 encoding tests."""
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    return img


@pytest.fixture
def api_parameters():
    """Mock API parameters."""
    return {
        "model": "test-model",
        "temperature": 0.7,
        "max_tokens": 512
    }


# --- Tests for parse_vlm_response --------------------------------------------

def test_parse_vlm_response_with_think_tags():
    """Test parsing VLM response with <think> tags."""
    response_text = "<think>Let me analyze this...</think>\n\nTurn left at the intersection."
    
    result = parse_vlm_response(response_text)
    
    assert result == "Turn left at the intersection."


def test_parse_vlm_response_without_think_tags():
    """Test parsing VLM response without <think> tags."""
    response_text = "Continue straight ahead."
    
    result = parse_vlm_response(response_text)
    
    assert result == "Continue straight ahead."


def test_parse_vlm_response_strips_whitespace():
    """Test that parse_vlm_response strips leading/trailing whitespace."""
    response_text = "  <think>Thinking...</think>\n\n  Stop at the red light.  \n"
    
    result = parse_vlm_response(response_text)
    
    assert result == "Stop at the red light."


# NOTE: This test case should not happen in practice, but we test it for robustness.
def test_parse_vlm_response_multiple_think_tags():
    """Test parsing response with multiple <think> blocks (only splits on first)."""
    response_text = "<think>First thought</think>\n\nAnswer with </think> in text."
    
    result = parse_vlm_response(response_text)
    
    # Should only split on the first occurrence
    assert result == "Answer with </think> in text."


# --- Tests for encode_image_to_base64 ----------------------------------------

@patch("untitled.generator.vlm.Image.open")
def test_encode_image_to_base64(mock_image_open, mock_image_file):
    """Test encoding an image to base64 data URL."""
    mock_image_open.return_value.__enter__ = Mock(return_value=mock_image_file)
    mock_image_open.return_value.__exit__ = Mock(return_value=False)
    
    test_path = Path("/tmp/test_image.jpg")
    result = encode_image_to_base64(test_path)
    
    # Verify it's a data URL
    assert result.startswith("data:image/jpeg;base64,")
    
    # Verify Image.open was called with the correct path
    mock_image_open.assert_called_once_with(test_path)
    
    # Verify the result contains base64 encoded data
    base64_data = result.split(",")[1]
    assert len(base64_data) > 0
    # Verify it's valid base64
    try:
        base64.b64decode(base64_data)
    except Exception:
        pytest.fail("Result is not valid base64")


@patch("untitled.generator.vlm.Image.open")
def test_encode_image_to_base64_format(mock_image_open, mock_image_file):
    """Test that image is saved in JPEG format."""
    # Track the save call
    mock_buffered = io.BytesIO()
    
    def mock_save(buffered, format):
        # Simulate writing some data
        buffered.write(b"fake_jpeg_data")
    
    mock_image_file.save = Mock(side_effect=mock_save)
    mock_image_open.return_value.__enter__ = Mock(return_value=mock_image_file)
    mock_image_open.return_value.__exit__ = Mock(return_value=False)
    
    test_path = Path("/tmp/test_image.jpg")
    result = encode_image_to_base64(test_path)
    
    # Verify save was called with JPEG format
    mock_image_file.save.assert_called_once()
    call_args = mock_image_file.save.call_args
    assert call_args[1]["format"] == "JPEG"


# --- Tests for VLM class -----------------------------------------------------

@patch("untitled.generator.vlm.OpenAI")  # mock_openai_class
def test_vlm_init_default_parameters(mock_openai_class):
    """Test VLM initialization with default parameters."""
    mock_client = Mock()
    mock_openai_class.return_value = mock_client
    
    vlm = VLM("http://localhost:8000", "test-key")
    
    # Verify OpenAI client was created with correct parameters
    mock_openai_class.assert_called_once_with(
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    # Verify API parameters
    assert vlm.api_parameters == {}
    assert vlm.client == mock_client


@patch("untitled.generator.vlm.OpenAI")  # mock_openai_class
def test_vlm_init_with_parameters(mock_openai_class, api_parameters):
    """Test VLM initialization with custom API parameters."""
    mock_client = Mock()
    mock_openai_class.return_value = mock_client
    
    vlm = VLM("http://localhost:8000", "test-key", api_parameters)
    
    # Verify API parameters are stored
    assert vlm.api_parameters == api_parameters
    assert vlm.client == mock_client


@patch("untitled.generator.vlm.encode_image_to_base64")  # mock_encode
@patch("untitled.generator.vlm.OpenAI")  # mock_openai_class
def test_vlm_prompt_with_text_only(mock_openai_class, mock_encode, mock_openai_client):
    """Test VLM prompt with text only (no images)."""
    mock_openai_class.return_value = mock_openai_client
    
    vlm = VLM("http://localhost:8000", "test-key")
    system_text = "You are a helpful assistant."
    user_text = "What should I do?"
    
    result = vlm.prompt(system_text, user_text, None)
    
    # Verify the response text is extracted
    assert result == "<think>Reasoning...</think>\n\nFinal answer."
    
    # Verify chat completion was called
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    
    # Verify message structure
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": system_text}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == [{"type": "text", "text": user_text}]
    
    # Verify encode_image_to_base64 was not called
    mock_encode.assert_not_called()


@patch("untitled.generator.vlm.encode_image_to_base64")  # mock_encode
@patch("untitled.generator.vlm.OpenAI")  # mock_openai_class
def test_vlm_prompt_with_images(mock_openai_class, mock_encode, mock_openai_client):
    """Test VLM prompt with images and text."""
    mock_openai_class.return_value = mock_openai_client
    mock_encode.side_effect = ["data:image/jpeg;base64,image1data", "data:image/jpeg;base64,image2data"]
    
    vlm = VLM("http://localhost:8000", "test-key")
    system_text = "You are a helpful assistant."
    user_text = "Describe these images."
    image_paths = [Path("/tmp/image1.jpg"), Path("/tmp/image2.jpg")]
    
    result = vlm.prompt(system_text, user_text, image_paths)
    
    # Verify the response text is extracted
    assert result == "<think>Reasoning...</think>\n\nFinal answer."
    
    # Verify encode_image_to_base64 was called for each image
    assert mock_encode.call_count == 2
    mock_encode.assert_any_call(Path("/tmp/image1.jpg"))
    mock_encode.assert_any_call(Path("/tmp/image2.jpg"))
    
    # Verify message structure includes images
    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert len(user_content) == 3  # 2 images + 1 text
    
    # Check image content
    assert user_content[0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,image1data"}
    }
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,image2data"}
    }
    
    # Check text content
    assert user_content[2] == {"type": "text", "text": user_text}


@patch("untitled.generator.vlm.encode_image_to_base64")  # mock_encode
@patch("untitled.generator.vlm.OpenAI")  # mock_openai_class
def test_vlm_prompt_with_api_parameters(mock_openai_class, mock_encode, mock_openai_client, api_parameters):
    """Test that VLM prompt passes API parameters to OpenAI."""
    mock_openai_class.return_value = mock_openai_client
    
    vlm = VLM("http://localhost:8000", "test-key", api_parameters)
    vlm.prompt("System", "User", None)
    
    # Verify API parameters were passed to create
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args[1]["model"] == "test-model"
    assert call_args[1]["temperature"] == 0.7
    assert call_args[1]["max_tokens"] == 512


@patch("untitled.generator.vlm.encode_image_to_base64")  # mock_encode
@patch("untitled.generator.vlm.OpenAI")  # mock_openai_class
def test_vlm_prompt_with_none_text(mock_openai_class, mock_encode, mock_openai_client):
    """Test VLM prompt with None text (only images)."""
    mock_openai_class.return_value = mock_openai_client
    mock_encode.return_value = "data:image/jpeg;base64,imagedata"
    
    vlm = VLM("http://localhost:8000", "test-key")
    system_text = "You are a helpful assistant."
    image_paths = [Path("/tmp/image.jpg")]
    
    result = vlm.prompt(system_text, None, image_paths)
    
    # Verify the response
    assert result == "<think>Reasoning...</think>\n\nFinal answer."
    
    # Verify message structure
    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    
    user_content = messages[1]["content"]
    assert len(user_content) == 1  # Only image, no text
    assert user_content[0]["type"] == "image_url"


@patch("untitled.generator.vlm.encode_image_to_base64")  # mock_encode
@patch("untitled.generator.vlm.time.time")  # mock_time
@patch("untitled.generator.vlm.OpenAI")  # mock_openai_class
def test_vlm_prompt_measures_time(mock_openai_class, mock_time, mock_encode, mock_openai_client):
    """Test that VLM prompt measures elapsed time."""
    mock_openai_class.return_value = mock_openai_client
    # Return time for VLM timing and any other calls (like logging)
    mock_time.return_value = 100.0
    
    vlm = VLM("http://localhost:8000", "test-key")
    vlm.prompt("System", "User", None)
    
    # Verify time.time was called at least twice (start and end of timing)
    assert mock_time.call_count >= 2


@patch("untitled.generator.vlm.encode_image_to_base64")  # mock_encode
@patch("untitled.generator.vlm.OpenAI")  # mock_openai_class
def test_vlm_prompt_extracts_message_content(mock_openai_class, mock_encode):
    """Test that VLM prompt correctly extracts message content from response."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Custom response text"
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client
    
    vlm = VLM("http://localhost:8000", "test-key")
    result = vlm.prompt("System", "User", None)
    
    assert result == "Custom response text"

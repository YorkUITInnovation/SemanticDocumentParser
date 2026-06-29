import pytest
import logging
from unittest.mock import MagicMock, AsyncMock
from SemanticDocumentParser.element_parsers.image_captioner import image_captioner, get_base64
from ragflow_sdk import RAGFlow
import httpx

class DummyRAGFlow(RAGFlow):
    def __init__(self, api_key: str, host: str, port: int):
        super().__init__(api_key, host, port)

    def get_chunks(self, doc_id: str):
        pass

@pytest.fixture
def mock_ragflow_client():
    """Fixture for a mocked RAGFlow client."""
    mock_client = DummyRAGFlow(api_key="test", host="test", port=80)
    mock_client.get_chunks = MagicMock(return_value=[
        {"type": "image", "content": "This is an image caption."},
    ])
    return mock_client

@pytest.mark.asyncio
async def test_get_base64_success(mocker):
    mock_response = AsyncMock()
    mock_response.content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82' # Minimal PNG
    mocker.patch('httpx.AsyncClient.get', return_value=mock_response)
    
    metadata = {'image_url': 'http://example.com/image.png'}
    result = await get_base64(metadata)
    assert result is not None
    assert 'image_base64' in result
    assert result['image_mime_type'] == 'image/png'

@pytest.mark.asyncio
async def test_get_base64_invalid_image(mocker):
    mock_response = AsyncMock()
    mock_response.content = b'this is not an image'
    mocker.patch('httpx.AsyncClient.get', return_value=mock_response)

    metadata = {'image_url': 'http://example.com/not_an_image'}
    result = await get_base64(metadata)
    assert result is None


@pytest.mark.asyncio
async def test_get_base64_network_error_logs_warning_without_traceback(mocker, caplog):
    mocker.patch('httpx.AsyncClient.get', side_effect=httpx.ConnectError("All connection attempts failed"))

    metadata = {'image_url': 'http://example.com/unreachable.png'}
    with caplog.at_level(logging.WARNING):
        result = await get_base64(metadata)

    assert result is None
    assert "Failed to download an image for a file." in caplog.text
    assert "Traceback" not in caplog.text

@pytest.mark.asyncio
async def test_image_captioner_with_mocked_data(mock_ragflow_client):
    """
    Test the image_captioner with mocked data.
    """
    elements = [
        {"type": "Image", "element_id": "1", "text": "", "metadata": {"image_base64": "test", "image_mime_type": "image/png"}},
    ]
    doc_id = "test_doc_id"

    result = await image_captioner(elements, mock_ragflow_client, doc_id)

    assert len(result) > 0
    assert "auto_caption" in result[0]["metadata"]
    assert "DESCRIPTION" in result[0]["text"]

@pytest.mark.asyncio
async def test_image_captioner_unsupported_mime_types(mock_ragflow_client):
    elements = [
        {"type": "Image", "element_id": "1", "text": "", "metadata": {"image_base64": "test", "image_mime_type": "image/svg+xml"}},
        {"type": "Image", "element_id": "2", "text": "", "metadata": {"image_base64": "test", "image_mime_type": "application/pdf"}},
    ]
    doc_id = "test_doc_id"

    result = await image_captioner(elements, mock_ragflow_client, doc_id)
    assert len(result) == 0

@pytest.mark.asyncio
async def test_image_captioner_no_caption_from_ragflow(mock_ragflow_client):
    mock_ragflow_client.get_chunks = MagicMock(return_value=[])
    elements = [
        {"type": "Image", "element_id": "1", "text": "", "metadata": {"image_base64": "test", "image_mime_type": "image/png"}},
    ]
    doc_id = "test_doc_id"

    result = await image_captioner(elements, mock_ragflow_client, doc_id)
    assert len(result) == 1
    assert "auto_caption" not in result[0]["metadata"]

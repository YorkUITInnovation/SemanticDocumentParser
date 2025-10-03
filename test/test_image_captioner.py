
import pytest
from unittest.mock import MagicMock
from SemanticDocumentParser.element_parsers.image_captioner import image_captioner
from ragflow_sdk import RAGFlow

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

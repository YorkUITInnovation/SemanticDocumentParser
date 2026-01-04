from unstructured.documents.elements import NarrativeText
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import io
from SemanticDocumentParser.parser import SemanticDocumentParser
from ragflow_sdk import RAGFlow

class DummyRAGFlow(RAGFlow):
    def __init__(self, api_key: str, host: str, port: int):
        super().__init__(api_key, host, port)

    async def upload_file(self, dataset_id: str, file_path: str):
        pass

    def get_chunks(self, doc_id: str):
        pass

@pytest.fixture
def mock_ragflow_client():
    """Fixture for a mocked RAGFlow client."""
    mock_client = DummyRAGFlow(api_key="test", host="test", port=80)
    mock_client.upload_file = AsyncMock(return_value={"document_name": "test_doc_id"})
    mock_client.get_chunks = MagicMock(return_value=[
        {"type": "text", "content": "This is a text chunk."},
        {"type": "image", "content": "This is an image caption."},
        {"type": "table", "content": "This is a table chunk."},
    ])
    return mock_client


@patch("SemanticDocumentParser.parser.partition_auto", return_value=[
    NarrativeText(text="This is a long paragraph that needs to be split."),
    NarrativeText(text="This is another long paragraph that needs to be split."),
])
def test_aparse_with_mocked_data(mock_partition, mock_ragflow_client):
    """
    Test the aparse method with mocked data.
    """
    parser = SemanticDocumentParser(ragflow_client=mock_ragflow_client, dataset_id="test_dataset")
    document = io.BytesIO(b"This is a test document.")
    document_filename = "test.txt"

    # Since aparse is an async function, we need to run it in an event loop
    import asyncio
    result, stats = asyncio.run(parser.aparse(document, document_filename))

    assert len(result) > 0
    assert "element_parse_time" in stats
    assert "metadata_parse_time" in stats
    assert "paragraph_parse_time" in stats
    assert "list_parse_time" in stats
    assert "table_parse_time_strategy_1" in stats
    assert "table_parse_time_strategy_2" in stats
    assert "combine_window_time" in stats
    assert "image_caption_time" in stats

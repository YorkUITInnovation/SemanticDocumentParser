
import pytest
from unittest.mock import MagicMock
from SemanticDocumentParser.element_parsers.semantic_splitter import semantic_splitter
from unstructured.documents.elements import NarrativeText, Title
from ragflow_sdk import RAGFlow

class DummyRAGFlow(RAGFlow):
    def __init__(self, api_key: str, host: str, port: int):
        super().__init__(api_key, host, port)

    def upload_file(self, dataset_id: str, file_path: str):
        return {"doc_ids": ["test_doc_id"]}

    def get_chunks(self, doc_id: str):
        pass

@pytest.fixture
def mock_ragflow_client():
    """Fixture for a mocked RAGFlow client."""
    mock_client = DummyRAGFlow(api_key="test", host="test", port=80)
    mock_client.get_chunks = MagicMock(return_value=[
        {"type": "text", "content": "This is the first chunk."},
        {"type": "text", "content": "This is the second chunk."},
    ])
    return mock_client


async def test_semantic_splitter_with_mocked_data(mock_ragflow_client):
    """
    Test the semantic_splitter with mocked data.
    """
    elements = [
        Title(text="This is a title"),
        NarrativeText(text="This is a long paragraph that needs to be split."),
        NarrativeText(text="This is another long paragraph that needs to be split."),
    ]
    doc_id = "test_doc_id"

    result = await semantic_splitter(elements, mock_ragflow_client, doc_id)

    assert len(result) > 0
    # The number of elements should be greater than the original number of elements
    # because the semantic splitter should have split the paragraphs.
    assert len(result) > len(elements)

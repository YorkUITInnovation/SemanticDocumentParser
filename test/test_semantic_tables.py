
import pytest
from unittest.mock import MagicMock
from SemanticDocumentParser.element_parsers.semantic_tables import semantic_tables
from unstructured.documents.elements import Table, Title
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
        {"type": "table", "content": "This is a table chunk."},
    ])
    return mock_client


async def test_semantic_tables_with_mocked_data(mock_ragflow_client):
    """
    Test the semantic_tables with mocked data.
    """
    elements = [
        Title(text="This is a title"),
        Table(text="This is a table."),
    ]
    doc_id = "test_doc_id"

    result = await semantic_tables(elements, mock_ragflow_client, doc_id)

    assert len(result) > 0
    # The table element should be replaced by a narrative text element.
    assert not any(isinstance(element, Table) for element in result)
    assert any(isinstance(element, Title) for element in result)

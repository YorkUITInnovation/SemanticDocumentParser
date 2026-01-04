import pytest
from SemanticDocumentParser.element_parsers.metadata_parser import metadata_parser

# Mock element class for testing
class MockMetadata:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockElement:
    def __init__(self, text, metadata_dict):
        self.text = text
        self.metadata = MockMetadata(**metadata_dict)

def test_link_detection():
    """Test the hyperlink parsing fix with mock data"""
    element = MockElement(
        text="Content can be found at this link: Excel file with data and graphs",
        metadata_dict={
            "link_texts": ["Excel file with data and graphs"],
            "link_urls": ["https://u-york-eclass.catalyst-ca.net/mod/forum/view.php?id=3497166"],
            "links": None  # This is why the original parser failed
        }
    )

    # Apply the enhanced metadata parser
    metadata_parser([element])

    # Check if the fix worked
    expected_link = "[Excel file with data and graphs](https://u-york-eclass.catalyst-ca.net/mod/forum/view.php?id=3497166)"
    assert expected_link in element.text

def test_backward_compatibility():
    """Test that the fix doesn't break existing functionality"""
    # Test case: Element without any links
    element = MockElement(
        text="This is just regular text without any links",
        metadata_dict={}
    )

    original_text = element.text
    metadata_parser([element])

    assert element.text == original_text
import pandas as pd
from unstructured.documents.elements import Table, Title, NarrativeText, ElementMetadata

from SemanticDocumentParser.element_parsers.al_tables import (
    read_tables_bs4mp,
    read_tables,
    render_tables_add_to_nodes_text,
    parse_nodes_text_into_yeehaw_bonafide_elements,
    al_table_parser,
)

SIMPLE_TABLE_HTML = "<table><tr><td>Header 1</td><td>Header 2</td></tr><tr><td>Row 1, Cell 1</td><td>Row 1, Cell 2</td></tr></table>"
TABLE_WITH_LINK_HTML = "<table><tr><td><a href='http://example.com'>Link</a></td><td>Cell</td></tr></table>"


def test_read_tables_bs4mp():
    # Test with a simple HTML table
    dfs = read_tables_bs4mp(SIMPLE_TABLE_HTML)
    assert len(dfs) == 1
    assert isinstance(dfs[0], pd.DataFrame)
    assert dfs[0].shape == (2, 2)
    assert dfs[0].iloc[0, 0] == "Header 1"

    # Test with a table with links
    dfs = read_tables_bs4mp(TABLE_WITH_LINK_HTML)
    assert len(dfs) == 1
    assert dfs[0].iloc[0, 0] == "[Link] (http://example.com)"

    # Test with multiple tables
    dfs = read_tables_bs4mp(SIMPLE_TABLE_HTML + TABLE_WITH_LINK_HTML)
    assert len(dfs) == 2

    # Test with no tables
    dfs = read_tables_bs4mp("<p>No table here</p>")
    assert len(dfs) == 0


def test_read_tables():
    mock_element = Table(text="test")
    
    # Test with a Title element
    previous_elements = [Title(text="My Table Title")]
    dfs, titles = read_tables(SIMPLE_TABLE_HTML, mock_element, previous_elements)
    assert len(dfs) == 1
    assert len(titles) == 1
    assert titles[0] == "My Table Title"

    # Test with no Title element
    previous_elements = [NarrativeText(text="Some text")]
    dfs, titles = read_tables(SIMPLE_TABLE_HTML, mock_element, previous_elements)
    assert len(dfs) == 1
    assert len(titles) == 1
    assert titles[0] == "Untitled Table"


def test_render_tables_add_to_nodes_text():
    df = pd.DataFrame([["Header1", "Header2"], ["Data1", "Data2"]])
    titles = ["Test Title"]
    nodes_text = render_tables_add_to_nodes_text(titles, [df])
    assert len(nodes_text) == 1
    assert "Test Title" in nodes_text[0]
    assert "The following Header1: Data1 has" in nodes_text[0]


def test_parse_nodes_text_into_yeehaw_bonafide_elements():
    nodes_text = ["This is a test node."]
    mock_element = Table(text="test", metadata=ElementMetadata())
    elements = parse_nodes_text_into_yeehaw_bonafide_elements(nodes_text, mock_element)
    assert len(elements) == 1
    assert isinstance(elements[0], NarrativeText)
    assert elements[0].text == "This is a test node."
    assert "signature" in elements[0].metadata.to_dict()


def test_al_table_parser():
    # Test with a Table element
    elements = [
        Title(text="My Table Title"),
        Table(text="test", metadata=ElementMetadata(text_as_html=SIMPLE_TABLE_HTML)),
    ]
    parsed_elements = al_table_parser(elements)
    # Expect original elements + new NarrativeText element for the table
    assert len(parsed_elements) == 3
    assert isinstance(parsed_elements[0], Title)
    assert isinstance(parsed_elements[1], Table)
    assert isinstance(parsed_elements[2], NarrativeText)
    assert "My Table Title" in parsed_elements[2].text

    # Test with no Table element
    elements = [NarrativeText(text="Some text")]
    parsed_elements = al_table_parser(elements)
    assert len(parsed_elements) == 1
    assert elements[0] == parsed_elements[0]

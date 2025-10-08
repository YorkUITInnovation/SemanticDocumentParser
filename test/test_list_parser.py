from unstructured.documents.elements import ListItem, NarrativeText, Title, PageBreak, ElementMetadata

from SemanticDocumentParser.element_parsers.list_parser import (
    _iterate_without_page_breaks,
    _list_group_parser,
    list_parser,
)

def test_iterate_without_page_breaks():
    elements = [NarrativeText(text="1"), PageBreak(text=""), NarrativeText(text="2")]
    result = list(_iterate_without_page_breaks(elements))
    assert len(result) == 2
    assert not any(isinstance(el, PageBreak) for el in result)

def test_list_group_parser_short_list():
    list_items = [ListItem(text=f"item {i}") for i in range(5)]
    header = Title(text="My List")
    result = _list_group_parser(list_items, header)
    assert len(result) == 1
    assert isinstance(result[0], NarrativeText)
    assert "My List" in result[0].text
    assert "- item 0" in result[0].text

def test_list_group_parser_long_list():
    list_items = [ListItem(text=f"item {i}" * 100) for i in range(20)]
    header = Title(text="Long List")
    result = _list_group_parser(list_items, header)
    assert len(result) > 1
    assert all(isinstance(el, NarrativeText) for el in result)
    assert "Part 1" in result[0].text
    assert "Part 2" in result[1].text

def test_list_group_parser_no_header():
    list_items = [ListItem(text="item 1")]
    result = _list_group_parser(list_items, None)
    assert "Untitled" in result[0].text

def test_list_group_parser_header_with_depth():
    list_items = [ListItem(text="item 1")]
    header = Title(text="Deep List", metadata=ElementMetadata(category_depth=2))
    result = _list_group_parser(list_items, header)
    assert "### Deep List" in result[0].text

def test_list_parser_simple():
    elements = [
        Title(text="List Title"),
        ListItem(text="item 1"),
        ListItem(text="item 2"),
    ]
    result = list_parser(elements)
    assert len(result) == 2 # Title and NarrativeText
    assert isinstance(result[1], NarrativeText)
    assert "List Title" in result[1].text

def test_list_parser_interrupted():
    elements = [
        Title(text="List Title"),
        ListItem(text="item 1"),
        NarrativeText(text="interruption"),
        ListItem(text="item 2"),
    ]
    result = list_parser(elements)
    # The "interruption" is considered an intro to the next list and is removed.
    # The first list is flushed, creating a NarrativeText.
    # The second list is created, and at the end it is flushed, creating another NarrativeText.
    assert len(result) == 3 # Title, NarrativeText (from list 1), NarrativeText (from list 2)

def test_list_parser_with_intro():
    elements = [
        Title(text="List Title"),
        NarrativeText(text="This is an intro"),
        ListItem(text="item 1"),
    ]
    result = list_parser(elements)
    assert len(result) == 2 # Title and NarrativeText (intro is removed)
    assert "List Title" in result[1].text

def test_list_parser_with_page_break():
    elements = [
        ListItem(text="item 1"),
        PageBreak(text=""),
        ListItem(text="item 2"),
    ]
    result = list_parser(elements)
    assert len(result) == 1
    assert isinstance(result[0], NarrativeText)
    assert "- item 1" in result[0].text
    assert "- item 2" in result[0].text

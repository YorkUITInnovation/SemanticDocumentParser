from typing import Generator
from typing import List, Optional

from unstructured.documents.elements import Element, ListItem, NarrativeText, Title, PageBreak


def _list_group_parser(elements: List[ListItem], header_node: Optional[NarrativeText]) -> List[Element]:
    """
    Deconstruct a list group into semantic units grouped into chunks with labeled headers.

    :param elements: The elements in the group
    :param header_node: The node just prior to the ListItem list
    :return: The deconstructed list
    """

    nodes: List[Element] = []
    header_title: str = header_node.text if header_node else "Untitled"

    if (
            header_node and
            hasattr(header_node.metadata, 'category_depth') and
            header_node.metadata.category_depth is not None and
            isinstance(header_node.metadata.category_depth, int)
    ):
        prefix = "#" * header_node.metadata.category_depth
    else:
        prefix = "##"

    header_level: str = prefix + "#"

    header_sub_level = header_level + "#"
    full_list_text = "\n".join([f"- {element.text}" for element in elements])

    # If total text is short, return as a single NarrativeText node without group headers
    if len(full_list_text) < 750:
        return [NarrativeText(text=full_list_text)]

    group_number = 1
    current_group: List[str] = []
    current_length = 0

    def make_group_text(group_items: List[str], number: int) -> NarrativeText:
        group_header = f"{header_level} {header_title}\n\n{header_sub_level} Part {number}:\n\n"
        group_text = group_header + "\n".join(group_items)
        return NarrativeText(text=group_text)

    for element in elements:
        item_text = f"- {element.text}"
        if current_length + len(item_text) > 1500 and current_group:
            nodes.append(make_group_text(current_group, group_number))
            group_number += 1
            current_group = []
            current_length = 0
        current_group.append(item_text)
        current_length += len(item_text)

    if current_group:
        nodes.append(make_group_text(current_group, group_number))

    return nodes


def _iterate_without_page_breaks(elements: List[Element]) -> Generator[Element, None, None]:
    """
    Remove page breaks using a cheeky generator method. Necessary for list parser across multiple pages.

    :param elements: The elements to iterate through
    :return: None

    """

    for element in elements:
        if isinstance(element, PageBreak):
            continue
        yield element


def list_parser(elements: List[Element]) -> List[Element]:
    """
    Each item in a list is its own semantic unit of information.

    Lists should be represented in their TOTAL form, but also with individual items.

    :param elements: All elements of a document
    :return: Elements with lists nodes enhanced properly and converted to NarrativeText

    """

    nodes: List[Element] = []
    header_node: Optional[Element] = None

    list_group: List[ListItem] = []
    last_node: Optional[Element] = None

    for element in _iterate_without_page_breaks(elements):

        # If it's a list item then add it to the current group
        if isinstance(element, ListItem):
            list_group.append(element)

            # If the last node was text & now it's a list, set the header node
            if isinstance(last_node, NarrativeText) or isinstance(last_node, Title):
                header_node = last_node

        else:
            # If not a list item just add it directly
            nodes.append(element)

            # If the last node was a list node & now it isn't, run the parser
            if isinstance(last_node, ListItem):
                nodes.extend(_list_group_parser(list_group, header_node))
                list_group = []
                header_node = None

        last_node = element

    return nodes

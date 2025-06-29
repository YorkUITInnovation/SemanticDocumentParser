from typing import List

from unstructured.documents.elements import Element


def _parse_element_urls(element: Element) -> None:
    """
    Replace the URL in-text into the element using Markdown hyperlink syntax. In-place modification of element.

    Known Limitation: Unstructured does not parse the hyperlinks within Table elements.

    :param element: The element to parse
    :return: None
    """

    change_delta: int = 0

    for link in element.metadata.links:
        # Deconstruct dict
        start_index: int = link['start_index'] + change_delta
        link_text: str = link['text']
        end_index: int = start_index + len(link_text)

        # Create Markdown link
        markdown_link = f"[{link_text}]({link['url']})"
        change_delta += len(markdown_link) - len(link_text)
        element.text = element.text[:start_index] + markdown_link + element.text[end_index:]

    # Clean up unused metadata
    element.metadata.link_texts = None
    element.metadata.links = None
    element.metadata.link_urls = None


def metadata_parser(elements: List[Element]) -> None:
    """
    Extract hyperlinks and substitute them in natural language. In-place modification of array.
    Remove extra metadata fields that are unnecessary and annoying to debug with.

    :param elements: Element list
    :return: None

    """

    for element in elements:

        # Must have a link in the element
        if element.metadata.links:
            _parse_element_urls(element)

        # Other stuff we don't care about
        element.metadata.filetype = None
        element.metadata.languages = None
        element.metadata.page_number = None


__all__ = ["metadata_parser"]

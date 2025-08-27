import re
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


def _manual_link_detection(element: Element) -> None:
    """
    Manually detect and replace common link patterns that unstructured might miss.
    This handles cases where unstructured stores link info in link_texts/link_urls instead of links array.
    """
    text = element.text
    
    # Check if we have link texts and URLs in metadata
    if (hasattr(element.metadata, 'link_texts') and element.metadata.link_texts and
        hasattr(element.metadata, 'link_urls') and element.metadata.link_urls):
        
        # Match each link text with its corresponding URL
        for i, link_text in enumerate(element.metadata.link_texts):
            if i < len(element.metadata.link_urls):
                url = element.metadata.link_urls[i]
                markdown_link = f"[{link_text}]({url})"
                text = text.replace(link_text, markdown_link)
    
    # Also check for HTML in text_as_html metadata
    elif hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
        html = element.metadata.text_as_html
        href_matches = re.findall(r'<a[^>]*href="([^"]*)"[^>]*>([^<]*)</a>', html)
        for url, link_text in href_matches:
            markdown_link = f"[{link_text}]({url})"
            text = text.replace(link_text, markdown_link)
    
    element.text = text


def metadata_parser(elements: List[Element]) -> None:
    """
    Extract hyperlinks and substitute them in natural language. In-place modification of array.
    Remove extra metadata fields that are unnecessary and annoying to debug with.

    :param elements: Element list
    :return: None

    """

    for element in elements:
        # Check if links are detected in the standard format
        if hasattr(element.metadata, 'links') and element.metadata.links:
            _parse_element_urls(element)
        else:
            # Try manual link detection for cases unstructured stores differently
            _manual_link_detection(element)

        # Other stuff we don't care about
        element.metadata.filetype = None
        element.metadata.languages = None
        element.metadata.page_number = None
        
        # Clean up link metadata after processing
        if hasattr(element.metadata, 'link_texts'):
            element.metadata.link_texts = None
        if hasattr(element.metadata, 'link_urls'):
            element.metadata.link_urls = None


__all__ = ["metadata_parser"]

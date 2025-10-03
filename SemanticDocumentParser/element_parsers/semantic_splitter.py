import asyncio
import itertools
import os
import tempfile
from asyncio import Task
from typing import List, TypedDict, Optional, Tuple

from ragflow_sdk import RAGFlow
from unstructured.documents.elements import Element, Title, NarrativeText


class ElementGroup(TypedDict):
    """Groups of elements split by consecutive Title objects"""

    title_node: Optional[Title]
    nodes: List[Element]


def _create_element_groups(elements: List[Element]) -> List[ElementGroup]:
    """
    Create element groups between Title elements.

    Elements between Titles represent semantically different units of information, so we have a guaranteed
    semantic boundary we can exploit in chunking.

    :param elements: The element array
    :return: The grouped elements

    """

    element_groups: List[ElementGroup] = []
    current_group = ElementGroup(title_node=None, nodes=[])

    # Parse groups
    for element in elements:

        if isinstance(element, Title):
            if current_group:
                element_groups.append(current_group)

            current_group = ElementGroup(title_node=element, nodes=[])

        elif current_group is not None:
            current_group['nodes'].append(element)

    # Get rid of the remaining group
    if bool(current_group):
        element_groups.append(current_group)

    # If no groups, then return the original list
    return element_groups if element_groups else [ElementGroup(title_node=None, nodes=elements)]


async def _semantic_split_node(
        title_node: Optional[Title],
        node: NarrativeText,
        ragflow_client: RAGFlow,
        dataset_id: str
) -> List[NarrativeText]:
    """
    Run semantic splitting on each text node to subdivide bulky paragraphs into semantic units

    :param title_node: The Title the node falls under
    :param node: The node to parse
    :param ragflow_client: The RAGFlow client
    :param dataset_id: The ID of the RAGFlow dataset
    :return: The unstructured NarrativeText elements

    """

    elements: List[NarrativeText] = []
    header_level: str = ("#" * (title_node.metadata.category_depth or 2)) if title_node and hasattr(title_node.metadata, 'category_depth') else "##"
    title_text: str = (header_level + " " + title_node.text) if title_node else ""

    # Create a temporary file to upload to RAGflow
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file.write(node.text)
        temp_file_path = temp_file.name

    try:
        # Upload the file to RAGflow in a separate thread
        upload_result = await asyncio.to_thread(ragflow_client.upload_file, dataset_id, temp_file_path)
        doc_id = upload_result['doc_ids'][0]

        # Get the chunks from RAGflow in a separate thread
        chunks = await asyncio.to_thread(ragflow_client.get_chunks, doc_id)

        # Regenerate NarrativeText elements
        for chunk in chunks:
            elements.append(
                NarrativeText(
                    # The title node may be important to describe the node contents
                    text=title_text + "\n" + chunk['content'],
                    metadata=node.metadata
                )
            )
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

    return elements


PARSER_GENERATED_SIGNATURE = "PARSER_GENERATED"


async def _semantic_split_element_group(
        group: ElementGroup,
        ragflow_client: RAGFlow,
        dataset_id: str
):
    """
    Process an element group. Semantically split paragraphs into further nodes.

    :param group: The element group to process
    :param ragflow_client: The RAGFlow client
    :param dataset_id: The ID of the RAGFlow dataset
    :return: The 1D processed node split

    """

    # First represent the entire group itself, in case that provides more complete meaning
    nodes: List[Element] = []
    # preserve the heading Title at the start of this group
    if group['title_node'] is not None:
        nodes.append(group['title_node'])

    parse_tasks: List[Task] = []

    for node in group['nodes']:

        # Other node types can be parsed as their own semantic units & just need to be passed on
        if (not isinstance(node, NarrativeText)) or node.metadata.data_source == "GENERATED":
            nodes.append(node)
            continue

        parse_tasks.append(
            asyncio.create_task(
                _semantic_split_node(
                    group['title_node'],
                    node,
                    ragflow_client,
                    dataset_id
                )
            )
        )

    parse_result: Tuple[List[Element]] = await asyncio.gather(*parse_tasks)
    nodes.extend(list(itertools.chain.from_iterable(parse_result)))
    return nodes


async def semantic_splitter(
        elements: List[Element],
        ragflow_client: RAGFlow,
        dataset_id: str
) -> List[Element]:
    """

    Re-distribute NarrativeTexts as chunks based on semantic similarity of adjacent texts.

    The process roughly follows:
        1. Group by title elements
        2. Run semantic splitting within each group
            i. By grouping consecutive NarrativeTexts
            ii. By running semantic splitting WITHIN these chunks
            iii. By returning a 1D array for each group that gets combined

    Edge Cases Handled:
        - Adjacent titles

    :param ragflow_client: The RAGFlow client
    :param dataset_id: The ID of the RAGFlow dataset
    :param elements: All elements in the document
    :return: The new list of elements with relationships respected

    """

    # Split into groups between Title elements
    element_groups: List[ElementGroup] = _create_element_groups(elements)
    parse_tasks: List[Task] = []

    for group in element_groups:
        parse_tasks.append(
            asyncio.create_task(
                _semantic_split_element_group(
                    group, ragflow_client, dataset_id
                )
            )
        )

    parse_result: Tuple[List[Element]] = await asyncio.gather(*parse_tasks)
    return list(itertools.chain.from_iterable(parse_result))

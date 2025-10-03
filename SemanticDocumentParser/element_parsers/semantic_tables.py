import asyncio
import json
import logging
import textwrap
import traceback
from json import JSONDecodeError
from typing import List, Awaitable, Optional, Union

from ragflow_sdk import RAGFlow
from unstructured.documents.elements import Element, Table, NarrativeText, Title


async def _semantic_ingest_table(
        element: Table,
        previous_element: Optional[Union[NarrativeText, Title]],
        ragflow_client: RAGFlow,
        doc_id: str
) -> List[NarrativeText]:
    """
    Parse the table & create a summary of it. Also include a raw copy of the table.

    :param element: The element to parse
    :param previous_element: The previous element before the table, if it was a Title or NarrativeText
    :param ragflow_client: The RAGFlow client
    :param doc_id: The ID of the document in RAGFlow
    :return: List of NarrativeText elements generated from the table

    """

    elements: List[NarrativeText] = []

    # Get the chunks from RAGflow in a separate thread
    chunks = await asyncio.to_thread(ragflow_client.get_chunks, doc_id)
    table_chunks = [chunk for chunk in chunks if chunk.get('type') == 'table']

    # Find the corresponding table chunk from RAGflow
    # This is a bit of a hack, we are assuming the tables are in the same order
    if table_chunks:
        table_chunk = table_chunks.pop(0)
        parsed_text = table_chunk.get('content', '')
        element_header: str = previous_element.text + "\n\n" if previous_element else ""
        elements.append(
            NarrativeText(
                text=element_header + parsed_text,
                metadata=element.metadata
            )
        )

    return elements


async def semantic_tables(elements: List[Element], ragflow_client: RAGFlow, doc_id: str) -> List[Element]:
    """
    Semantically separate tables into natural language using an LLM

    [WARNING: CONSUMES THE TABLE, SO IT IS NO LONGER AN ELEMENT. RUN THIS LAST.]

    :param elements: The elements in the table
    :param ragflow_client: The RAGFlow client
    :param doc_id: The ID of the document in RAGFlow
    :return: The list of elements parsed from the table

    """

    tasks: List[Awaitable] = []
    nodes: List[Element] = []

    # Create the comprehension tasks
    for idx, element in enumerate(elements):

        if not isinstance(element, Table):
            nodes.append(element)
            continue

        previous_element: Optional[Element] = None

        # Only include if the previous node is a TITLE or TEXT element
        if idx > 0:
            if isinstance(elements[idx - 1], NarrativeText) or isinstance(elements[idx - 1], Title):
                previous_element = elements[idx - 1]

        # Add the task
        tasks.append(
            _semantic_ingest_table(
                element, previous_element, ragflow_client, doc_id
            )
        )

    # Return the list as a 1D array
    for item in await asyncio.gather(*tasks):
        if isinstance(item, list):
            nodes.extend(item)
        else:
            nodes.append(item)

    return nodes

import functools
import io
from typing import List, Tuple, TypedDict, Optional

from llama_index.core.llms import LLM
from pydantic.v1 import BaseModel
from unstructured.partition.auto import partition

from SemanticDocumentParser.element_parsers.list_parser import list_parser
from SemanticDocumentParser.element_parsers.metadata_parser import metadata_parser
from SemanticDocumentParser.element_parsers.remove_small import remove_small
from SemanticDocumentParser.element_parsers.semantic_splitter import semantic_splitter
from SemanticDocumentParser.element_parsers.semantic_tables import semantic_tables
from SemanticDocumentParser.element_parsers.window_parser import window_parser
from SemanticDocumentParser.llama_extensions.node_parser import AsyncSemanticSplitterNodeParser
from SemanticDocumentParser.utils import with_timings_sync, with_timings_async


class SemanticDocumentParserStats(TypedDict):
    element_parse_time: Optional[float]
    metadata_parse_time: Optional[float]
    paragraph_parse_time: Optional[float]
    list_parse_time: Optional[float]
    table_parse_time: Optional[float]
    combine_window_time: Optional[float]


class SemanticDocumentParser(BaseModel):
    """
    Split nodes into semantic units

    """

    llm_model: LLM
    node_parser: AsyncSemanticSplitterNodeParser

    async def aparse(
            self,
            document: io.BytesIO,
            document_filename: str
    ) -> Tuple[List[dict], SemanticDocumentParserStats]:
        """
        Asynchronously (where possible) parse the document

        :param document: The document to parse of any type unstructured supports
        :param document_filename: The name of the doc
        :return: A list of elements existing as distinct chunks of NarrativeText

        """

        # Generate the document-agnostic array
        element_parse_time, elements = with_timings_sync(
            fn=functools.partial(
                partition,
                file=document,
                metadata_filename=document_filename
            )
        )

        # If there are no elements, don't run the parsers
        if len(elements) < 1:
            stats: SemanticDocumentParserStats = {
                "element_parse_time": element_parse_time,
                "metadata_parse_time": None,
                "paragraph_parse_time": None,
                "list_parse_time": None,
                "table_parse_time": None,
                "combine_window_time": None
            }

            return [], stats

        # Parse document metadata
        metadata_parse_time, _ = with_timings_sync(fn=functools.partial(metadata_parser, elements))

        # Group elements by title separation, then split unrelated texts into smaller ones
        paragraph_parse_time, elements = await with_timings_async(semantic_splitter(elements, self.node_parser))

        # Group the list items into individual nodes
        list_parse_time, elements = with_timings_sync(fn=functools.partial(list_parser, elements))

        # Combination A) Combine nodes naively with the Window approach (smaller nodes)
        # MUST DO THIS FIRST. Combo B will alter nodes IN-PLACE!!!
        combine_window_time, dict_elements_2 = with_timings_sync(
            fn=functools.partial(window_parser, elements)
        )

        # Parse tables
        table_parse_time, elements = await with_timings_async(
            semantic_tables(
                elements,
                self.llm_model
            )
        )

        return (
            remove_small(dict_elements_2, min_length=10),
            SemanticDocumentParserStats(
                element_parse_time=element_parse_time,
                metadata_parse_time=metadata_parse_time,
                paragraph_parse_time=paragraph_parse_time,
                list_parse_time=list_parse_time,
                table_parse_time=table_parse_time,
                combine_window_time=combine_window_time,
            )
        )


__all__ = ['SemanticDocumentParser']

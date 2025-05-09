import asyncio
import functools
import io
from typing import List, Tuple, TypedDict, Optional, Awaitable, Callable

from llama_index.core.node_parser import NodeParser
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from pydantic.v1 import BaseModel
from unstructured.file_utils.filetype import detect_filetype
from unstructured.file_utils.model import FileType
from unstructured.partition.auto import partition as partition_auto
from unstructured_expanded.partition.docx import partition_docx
from unstructured_expanded.partition.pdf import partition_pdf
from unstructured_expanded.partition.pptx.partition_pptx import partition_pptx

from SemanticDocumentParser.element_parsers.al_tables import al_table_parser
from SemanticDocumentParser.element_parsers.image_captioner import image_captioner
from SemanticDocumentParser.element_parsers.list_parser import list_parser
from SemanticDocumentParser.element_parsers.metadata_parser import metadata_parser
from SemanticDocumentParser.element_parsers.remove_small import remove_small
from SemanticDocumentParser.element_parsers.semantic_splitter import semantic_splitter
from SemanticDocumentParser.element_parsers.semantic_tables import semantic_tables
from SemanticDocumentParser.element_parsers.window_parser import window_parser
from SemanticDocumentParser.utils import with_timings_sync, with_timings_async


class SemanticDocumentParserStats(TypedDict):
    element_parse_time: Optional[float]
    metadata_parse_time: Optional[float]
    paragraph_parse_time: Optional[float]
    list_parse_time: Optional[float]
    table_parse_time_strategy_1: Optional[float]
    table_parse_time_strategy_2: Optional[float]
    combine_window_time: Optional[float]
    image_caption_time: Optional[float]


PARSER_OVERRIDE_MAP: dict[FileType, Callable] = {
    FileType.DOCX: partition_docx,
    FileType.PDF: partition_pdf,
    FileType.PPTX: partition_pptx,
}


class SemanticDocumentParser(BaseModel):
    """
    Split nodes into semantic units

    """

    llm_model: OpenAIMultiModal
    node_parser: NodeParser

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def partition(cls, **kwargs):
        # Detect the encoding
        file_type: FileType = detect_filetype(
            file=kwargs.get("file"),
            encoding=kwargs.get("encoding"),
            content_type=kwargs.get("content_type"),
            metadata_file_path=kwargs.get("metadata_filename"),
        )

        # Get the partition function from our definitions
        partition_fn = PARSER_OVERRIDE_MAP.get(file_type, partition_auto)

        # Otherwise, default the basic partitioning for other types
        return functools.partial(partition_fn, **kwargs)

    async def aparse(
            self,
            document: io.BytesIO,
            document_filename: str,
            on_step_finished: Callable[[str, float], Awaitable[None]] = lambda x, y: asyncio.sleep(0)
    ) -> Tuple[List[dict], SemanticDocumentParserStats]:
        """
        Asynchronously (where possible) parse the document

        :param document: The document to parse of any type unstructured supports
        :param document_filename: The name of the doc
        :param on_step_finished: A callback to call when a step is finished
        :return: A list of elements existing as distinct chunks of NarrativeText

        """

        # Generate the document-agnostic array
        element_parse_time, elements = with_timings_sync(
            fn=self.partition(
                # Note: Do NOT specify 'encoding' or 'content_type' here, it's auto-determined
                file=document,
                metadata_filename=document_filename,
                languages=["en", "fr"],
                xml_keep_tags=True
            )
        )

        await on_step_finished('Unstructured Partition', element_parse_time)

        # If there are no elements, don't run the parsers
        if len(elements) < 1:
            stats: SemanticDocumentParserStats = {
                "element_parse_time": element_parse_time,
                "metadata_parse_time": None,
                "paragraph_parse_time": None,
                "list_parse_time": None,
                "table_parse_time_strategy_1": None,
                "table_parse_time_strategy_2": None,
                "combine_window_time": None,
                "image_caption_time": None,
            }

            return [], stats

        # Parse document metadata
        metadata_parse_time, _ = with_timings_sync(fn=functools.partial(metadata_parser, elements))
        await on_step_finished('Metadata Parsing', metadata_parse_time)

        # Parse tables strategy 1 [DOES NOT CONSUME TABLE ELEMENTS]
        # Must occur BEFORE the semantic splitter
        table_parse_time_strategy_1, elements = with_timings_sync(
            fn=functools.partial(al_table_parser, elements)
        )

        await on_step_finished('Table Parsing 1/2', table_parse_time_strategy_1)

        # Group the list items into individual nodes
        list_parse_time, elements = with_timings_sync(fn=functools.partial(list_parser, elements))

        await on_step_finished('List Parsing', list_parse_time)

        # Group elements by title separation, then split unrelated texts into smaller ones
        # Note that the way grouping is set up, the auto-caption will be used in the 'Title' element since these descriptions
        # tend to be longer & we don't want to pollute
        paragraph_parse_time, elements = await with_timings_async(semantic_splitter(elements, self.node_parser))

        await on_step_finished('Paragraph Parsing', paragraph_parse_time)

        # Parse tables strategy 2 [CONSUMES TABLE ELEMENTS]
        table_parse_time_strategy_2, elements = await with_timings_async(
            semantic_tables(
                elements,
                self.llm_model
            )
        )

        await on_step_finished('Table Parsing 2/2', table_parse_time_strategy_2)

        # Caption images
        image_caption_time, dict_elements = await with_timings_async(
            image_captioner(
                [element.to_dict() for element in elements],
                self.llm_model
            )
        )

        await on_step_finished('Image Captioning', image_caption_time)

        # Combine nodes naively with the Window approach (smaller nodes)
        combine_window_time, dict_elements = with_timings_sync(
            fn=functools.partial(window_parser, dict_elements)
        )

        await on_step_finished('Window Combination', combine_window_time)

        dict_elements = remove_small(
            dict_elements,
            min_length=10
        )

        await on_step_finished('Remove Small Nodes', 0)

        return (
            dict_elements,
            SemanticDocumentParserStats(
                element_parse_time=element_parse_time,
                metadata_parse_time=metadata_parse_time,
                paragraph_parse_time=paragraph_parse_time,
                list_parse_time=list_parse_time,
                table_parse_time_strategy_1=table_parse_time_strategy_1,
                table_parse_time_strategy_2=table_parse_time_strategy_2,
                combine_window_time=combine_window_time,
                image_caption_time=image_caption_time
            )
        )


__all__ = ['SemanticDocumentParser']

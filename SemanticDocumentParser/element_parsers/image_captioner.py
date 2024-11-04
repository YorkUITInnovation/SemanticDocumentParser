from typing import List

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


async def image_captioner(elements: List[dict], llm: OpenAIMultiModal) -> List[dict]:
    """
    Caption images using the LLM.

    """

    # 3-series models do not support image input
    if 'gpt-3' in llm.metadata.model_name:
        return elements

    for element in elements:

        if element['type'] != 'Image':
            continue

        image_document = ImageDocument(
            image=element['metadata']['image_base64'],
            image_mimetype="image/jpeg"
        )

        response: CompletionResponse = await llm.acomplete(
            prompt=(
                "You are an agent part of a RAG pipeline. You will be given a single image. Your job is to describe everything in the image. "
                "If the image contains math formulae, you should write out those formulae in plain text. Whatever text you reply with will be used "
                "directly as a text element in a vector database as part of a RAG pipeline, so optimize your description for RAG. Avoid using phrases like "
                "'This is a picture of' or 'This image shows'. Instead, describe the image directly. "
                "Focus entirely on what is depicted, using simple, direct language optimized for retrieval."
            ),
            image_documents=[image_document],
        )

        element['metadata']['auto_caption'] = element['text']
        element['text'] = f"[IMAGE {element['element_id']} DESCRIPTION START]{response.text}[IMAGE {element['element_id']} DESCRIPTION END]"

    return elements

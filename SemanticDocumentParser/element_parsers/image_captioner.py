import base64
import io
import logging
import traceback
from typing import List, Optional

import httpx
import puremagic
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


async def get_base64(metadata: dict) -> Optional[str]:
    try:

        async with httpx.AsyncClient() as client:

            # Download the image from the URL
            if 'image_url' not in metadata:
                return None

            # Get the image data
            response = await client.get(
                metadata['image_url'],
                timeout=10,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
            )

            # Get as bytes; check that it is a valid image otherwise we risk injecting malware
            image_bytes = response.content
            stream = io.BytesIO(image_bytes)
            magic_value = puremagic.magic_stream(stream)[0]

            # Not a valid image
            if 'image' not in magic_value.mime_type or magic_value.confidence < 0.7:
                return None

            # Return b64
            return base64.b64encode(image_bytes).decode('utf-8')

    except:
        logging.warning("Failed to download an image for a file. This can most likely be ignored.", traceback.format_exc())
        return None


async def image_captioner(elements: List[dict], llm: OpenAIMultiModal) -> List[dict]:
    """
    Caption images using the LLM.

    """

    # 3-series models do not support image input
    if 'gpt-3' in llm.metadata.model_name:
        return elements

    for element in elements:

        if element['type'] != 'Image' or 'metadata' not in element:
            continue

        if 'image_url' in element['metadata']:
            element['metadata']['image_base64'] = await get_base64(element['metadata'])
            if not element['metadata']['image_base64']:
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

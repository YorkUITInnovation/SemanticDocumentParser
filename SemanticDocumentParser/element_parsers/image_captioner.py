import base64
import io
import logging
import re
import traceback
from typing import List

import httpx
import puremagic
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


def extract_base64_from_data_uri(data_uri: str) -> dict | None:
    """
    Extract base64 data and mime type from a data URI like 'data:image/png;base64,iVBOR...'
    """
    try:
        # Match data URI pattern: data:mime_type;base64,data
        match = re.match(r'^data:([^;]+);base64,(.+)$', data_uri)
        if not match:
            return None

        mime_type, base64_data = match.groups()

        # Validate it's an image mime type
        if not mime_type.startswith('image/'):
            return None

        # Validate base64 data by trying to decode it
        try:
            decoded_data = base64.b64decode(base64_data)
            # Validate it's actually an image using puremagic
            stream = io.BytesIO(decoded_data)
            magic_value = puremagic.magic_stream(stream)[0]

            if 'image' not in magic_value.mime_type or magic_value.confidence < 0.7:
                return None

        except Exception:
            return None

        return {
            'image_base64': base64_data,
            'image_mime_type': mime_type,
        }

    except Exception:
        logging.warning("Failed to extract base64 from data URI", traceback.format_exc())
        return None


async def get_base64(metadata: dict) -> dict | None:
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
            return {
                'image_base64': base64.b64encode(image_bytes).decode('utf-8'),
                'image_mime_type': magic_value.mime_type,
            }

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

        # Handle images with URLs
        if 'image_url' in element['metadata']:
            base64_data = await get_base64(element['metadata'])
            if base64_data:
                element['metadata'] = {**element['metadata'], **base64_data}

        # Handle inline base64 images from data URIs
        elif 'image_src' in element['metadata'] and element['metadata']['image_src'].startswith('data:'):
            base64_data = extract_base64_from_data_uri(element['metadata']['image_src'])
            if base64_data:
                element['metadata'] = {**element['metadata'], **base64_data}

        # Skip if we don't have base64 image data
        if 'image_base64' not in element['metadata']:
            continue

        image_document = ImageDocument(
            image=element['metadata']['image_base64'],
            image_mimetype=element['metadata'].get('image_mime_type', "image/jpeg")
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

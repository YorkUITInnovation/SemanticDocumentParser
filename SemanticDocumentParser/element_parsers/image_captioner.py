import base64
import io
import logging
import traceback
from typing import List
import asyncio

import httpx
import puremagic
from ragflow_sdk import RAGFlow


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
        logging.warning("Failed to download an image for a file. This can most likely be ignored.", exc_info=True)
        return None


async def image_captioner(elements: List[dict], ragflow_client: RAGFlow, doc_id: str) -> List[dict]:
    """
    Caption images using the LLM.

    """

    # Get the chunks from RAGflow in a separate thread
    chunks = await asyncio.to_thread(ragflow_client.get_chunks, doc_id)
    image_chunks = [chunk for chunk in chunks if chunk.get('type') == 'image']

    # Filter out SVG images and other unsupported elements first
    filtered_elements = []

    for element in elements:
        # Keep non-image elements as-is
        if element['type'] != 'Image' or 'metadata' not in element:
            filtered_elements.append(element)
            continue

        # Handle images with URLs (download and detect MIME type)
        if 'image_url' in element['metadata']:
            download_result = await get_base64(element['metadata'])
            if download_result:
                element['metadata'] = {**element['metadata'], **download_result}

            # If download failed, remove the element completely (no alt text preservation)
            if 'image_base64' not in element['metadata']:
                logging.warning(f"Failed to download image {element.get('element_id', 'unknown')}, removing from processing")
                continue

        # Skip if no image data is available
        if 'image_base64' not in element['metadata']:
            logging.warning(f"Image element {element.get('element_id', 'unknown')} has no image data, removing from processing")
            continue

        # Get and validate base64 data
        base64_data = element['metadata']['image_base64']
        if not base64_data or (isinstance(base64_data, str) and not base64_data.strip()):
            logging.warning(f"Image element {element.get('element_id', 'unknown')} has empty/invalid base64 data, removing from processing")
            continue

        # Use the detected MIME type if available, otherwise fallback to jpeg
        mime_type = element['metadata'].get('image_mime_type', 'image/jpeg')

        # Normalize and validate mime type
        if isinstance(mime_type, str):
            mime_type = mime_type.lower().strip()

        # Skip images with missing, None, or invalid mime type
        if not mime_type or mime_type is None or 'image_mime_type' not in element['metadata']:
            logging.warning(f"Image element {element.get('element_id', 'unknown')} has missing/invalid mime type, removing from processing")
            continue

        # Validate mime type format (must start with 'image/')
        if not mime_type.startswith('image/'):
            logging.warning(f"Image element {element.get('element_id', 'unknown')} has invalid mime type format '{mime_type}', removing from processing")
            continue

        # Skip SVG files as they're not supported by vision models
        if mime_type == 'image/svg+xml':
            logging.warning(f"Removing SVG image {element.get('element_id', 'unknown')} - not supported by vision models")
            continue

        # Update the normalized mime type back to metadata
        element['metadata']['image_mime_type'] = mime_type

        # Keep supported image elements
        filtered_elements.append(element)

    # Now process the filtered elements for captioning
    for element in filtered_elements:
        if element['type'] != 'Image' or 'metadata' not in element:
            continue

        # Find the corresponding image chunk from RAGflow
        # This is a bit of a hack, we are assuming the images are in the same order
        if image_chunks:
            image_chunk = image_chunks.pop(0)
            caption = image_chunk.get('content', '')
            element['metadata']['auto_caption'] = element['text']
            element['text'] = f"[IMAGE {element['element_id']} DESCRIPTION START]{caption}[IMAGE {element['element_id']} DESCRIPTION END]"

    return filtered_elements


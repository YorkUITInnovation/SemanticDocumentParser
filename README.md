# SemanticDocumentParser

A Python library for intelligently parsing and chunking various document formats (like `.docx`, `.pdf`, and `.pptx`) into a structured format suitable for Retrieval-Augmented Generation (RAG) pipelines.

This library uses the `unstructured` package for initial document partitioning and a `RAGFlow` backend to perform advanced processing like semantic text splitting and image captioning.

## Features

-   Parses a wide variety of file types supported by `unstructured`.
-   Uses semantic splitting to break down long paragraphs into coherent chunks.
-   Intelligently groups list items.
-   Parses tables using multiple strategies.
-   Can generate captions for images embedded within documents.
-   Combines small nodes into larger, more contextually complete chunks.

## Installation

For local development, clone the repository and install the package in editable mode:

```sh
pip install -e .
```

This will install all the necessary dependencies listed in `setup.py`.

## Usage

Here is a basic example of how to use the `SemanticDocumentParser` to parse a document.

```python
import asyncio
import io

# You will need a RAGFlow client from the CriadexSDK
# This is a placeholder for the actual SDK client
from ragflow_sdk import RAGFlow as RAGFlowClient

from SemanticDocumentParser import SemanticDocumentParser


async def main():
    # 1. Initialize the RAGFlow client (from CriadexSDK)
    # This client handles communication with the RAGflow backend.
    ragflow_client = RAGFlowClient(base_url="http://localhost:8080")

    # 2. Initialize the parser
    # Provide the RAGFlow client and a dataset ID.
    parser = SemanticDocumentParser(
        ragflow_client=ragflow_client,
        dataset_id="my_dataset_123"
    )

    # 3. Open your document file in binary mode
    file_path = "path/to/your/document.docx"
    file_name = "document.docx"

    with open(file_path, "rb") as f:
        # Use io.BytesIO to handle the file in memory
        file_in_memory = io.BytesIO(f.read())

        # 4. Parse the document
        # The `aparse` method is asynchronous.
        parsed_elements, stats = await parser.aparse(
            document=file_in_memory,
            document_filename=file_name
        )

        # 5. Print the results
        print(f"Successfully parsed {len(parsed_elements)} elements.")
        print("Parsing statistics:", stats)
        # print(parsed_elements)


if __name__ == "__main__":
    asyncio.run(main())

```

## Local Development & Testing

1.  **Create a virtual environment:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```sh
    # Install the library in editable mode and its dev dependencies
    pip install -e ".[dev]"
    ```

3.  **Run tests:**
    ```sh
    pytest
    ```

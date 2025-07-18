import setuptools

# PyPi upload Command
# rm -r dist ; python setup.py sdist ; python -m twine upload dist/*

manifest: dict = {
    "name": "SemanticDocumentParser",
    "license": "MIT",
    "author": "Isaac Kogan",
    "version": "0.1.9",
    "email": "info@isaackogan.com"
}

if __name__ == '__main__':
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        name=manifest["name"],
        packages=setuptools.find_packages(),
        version=manifest["version"],
        license=manifest["license"],
        author=manifest["author"],
        author_email=manifest["email"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=[
            "llama-index-core==0.11.21",
            "llama-index-llms-azure-openai==0.2.2",
            "llama-index-multi-modal-llms-azure-openai==0.2.0",
            "llama-index-embeddings-azure-openai==0.2.5",
            "bs4",
            "pandas",
            "unstructured[all-docs]==0.17.2",
            "unstructured_expanded==0.17.2",
            "numpy==1.26.4",
            "httpx",
            "puremagic==1.30"
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Software Development :: Build Tools",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ]
    )

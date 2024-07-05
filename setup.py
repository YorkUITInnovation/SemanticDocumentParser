import setuptools

# PyPi upload Command
# rm -r dist ; python setup.py sdist ; python -m twine upload dist/*

manifest: dict = {
    "name": "SemanticDocumentParser",
    "license": "MIT",
    "author": "Isaac Kogan",
    "version": "0.0.7",
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
            "llama-index-core",
            "llama-index-llms-azure-openai",
            "llama-index-embeddings-azure-openai",
            "unstructured[all-docs]==0.14.6",
            "numpy==1.26.4"
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Software Development :: Build Tools",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ]
    )

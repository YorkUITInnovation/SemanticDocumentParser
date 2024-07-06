from typing import Sequence, List

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, Document


class AsyncSemanticSplitterNodeParser(SemanticSplitterNodeParser):

    async def abuild_semantic_nodes_from_documents(
            self,
            documents: Sequence[Document],
            show_progress: bool = False,
    ) -> List[BaseNode]:
        """Build window nodes from documents.
        <Same as regular but uses async embeddings call>
        """

        all_nodes: List[BaseNode] = []
        for doc in documents:
            text = doc.text
            text_splits = self.sentence_splitter(text)

            sentences = self._build_sentence_groups(text_splits)

            combined_sentence_embeddings = await self.embed_model.aget_text_embedding_batch(
                [s["combined_sentence"] for s in sentences],
                show_progress=show_progress,
            )

            for i, embedding in enumerate(combined_sentence_embeddings):
                sentences[i]["combined_sentence_embedding"] = embedding

            distances = self._calculate_distances_between_sentence_groups(sentences)

            chunks = self._build_node_chunks(sentences, distances)

            nodes = build_nodes_from_splits(
                chunks,
                doc,
                id_func=self.id_func,
            )

            all_nodes.extend(nodes)

        return all_nodes

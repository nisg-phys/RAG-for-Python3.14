from rank_bm25 import BM25Okapi
import numpy as np
from ragbot.utils.logger import get_logger

logger = get_logger("hybrid_retriever")

"""This script takes the query and retrieves relevant documents using both vector similarity search and keyword-based BM25 search. It then merges the results to provide a comprehensive set of relevant documents for the query."""

class HybridRetriever:

    def __init__(self, vectorstore, documents):

        self.vectorstore = vectorstore

        self.documents = documents

        tokenized_docs = [doc.page_content.split() for doc in documents]

        self.bm25 = BM25Okapi(tokenized_docs)


    def keyword_search(self, query, k=5):

        logger.info(f"Keyword search started")

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        top_k = np.argsort(scores)[::-1][:k]

        docs = [(self.documents[i], float(scores[i])) for i in top_k]


        logger.info(f"Keyword search returned {len(docs)} documents")


        return docs


    def vector_search(self, query, k=5):

        logger.info("Vector search started")
        return self.vectorstore.vectorstore.similarity_search_with_score(query, k=k)


  # Use RRF (Reciprocal Rank Fusion) for proper merging
    def retrieve(self, query, k=5):
        logger.info(f"Hybrid retrieval started for query: '{query[:50]}...' with k={k}")
    
        # Perform searches
        vector_results = self.vector_search(query, k)
        keyword_results = self.keyword_search(query, k)
        logger.info(f"Vector docs: {len(vector_results)} | Keyword docs: {len(keyword_results)}")
    
        # Score by rank (RRF)
        logger.debug("Calculating RRF scores")
        scores = {}
        docs_by_chunk_id = {}
        vector_scores = {}
        bm25_scores = {}

        for rank, (doc, vector_score) in enumerate(vector_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content)
            docs_by_chunk_id[chunk_id] = doc
            vector_scores[chunk_id] = float(vector_score)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rank + 60)
            logger.debug(f"Vector rank {rank}: score += {1/(rank+60):.4f}")
    
        for rank, (doc, bm25_score) in enumerate(keyword_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content)
            docs_by_chunk_id[chunk_id] = doc
            bm25_scores[chunk_id] = float(bm25_score)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rank + 60)
            logger.debug(f"Keyword rank {rank}: score += {1/(rank+60):.4f}")
    
        logger.info(f"Total unique documents after merging: {len(docs_by_chunk_id)}")
    
        sorted_chunk_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
        final_results = [
            {
                "doc": docs_by_chunk_id[chunk_id],
                "vector_score": vector_scores.get(chunk_id, 0.0),
                "bm25_score": bm25_scores.get(chunk_id, 0.0),
                "rrf_score": scores[chunk_id],
            }
            for chunk_id in sorted_chunk_ids[:k]
        ]
    
        logger.info(f"Returning top {len(final_results)} documents")
        for i, result in enumerate(final_results, start=1):
            doc = result["doc"]
            logger.info(
                "Rank %s chunk_id=%s rrf_score=%.4f vector_score=%.4f bm25_score=%.4f preview=%s",
                i,
                doc.metadata.get("chunk_id", "unknown"),
                result["rrf_score"],
                result["vector_score"],
                result["bm25_score"],
                doc.page_content[:100],
            )
    
        return final_results

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

        docs = [self.documents[i] for i in top_k]


        logger.info(f"Keyword search returned {len(docs)} documents")


        return docs


    def vector_search(self, query, k=5):

        return self.vectorstore.similarity_search(query, k=k)


  # Use RRF (Reciprocal Rank Fusion) for proper merging
    def retrieve(self, query, k=5):
        logger.info(f"Hybrid retrieval started for query: '{query[:50]}...' with k={k}")
    
        # Perform searches
        vector_docs = self.vector_search(query, k)
        keyword_docs = self.keyword_search(query, k)
        logger.info(f"Vector docs: {len(vector_docs)} | Keyword docs: {len(keyword_docs)}")
    
        # Score by rank (RRF)
        logger.debug("Calculating RRF scores")
        scores = {}
        for rank, doc in enumerate(vector_docs):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1/(rank+60)
            logger.debug(f"Vector rank {rank}: score += {1/(rank+60):.4f}")
    
        for rank, doc in enumerate(keyword_docs):
            scores[doc.page_content] = scores.get(doc.page_content, 0) + 1/(rank+60)
            logger.debug(f"Keyword rank {rank}: score += {1/(rank+60):.4f}")
    
    # Merge and deduplicate
        all_docs = {doc.page_content: doc for doc in vector_docs + keyword_docs}
        logger.info(f"Total unique documents after merging: {len(all_docs)}")
    
    # Sort by score
        sorted_docs = sorted(all_docs.items(), key=lambda x: scores[x[0]], reverse=True)
        final_docs = [doc for _, doc in sorted_docs[:k]]
    
        logger.info(f"Returning top {len(final_docs)} documents")
        for i, doc in enumerate(final_docs):
            logger.debug(f"Rank {i+1}: score={scores[doc.page_content]:.4f}, preview={doc.page_content[:100]}")
    
        return final_docs
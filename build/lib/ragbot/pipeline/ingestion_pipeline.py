from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbot.vectorstore.pinecone_store import PineconeStore


class IngestionPipeline:
    """
    Handles document chunking and ingestion into Pinecone.
    """

    def __init__(self):

        self.vectorstore = PineconeStore()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

    def ingest(self, documents):
        """
        Split documents into chunks and store them in Pinecone.
        """

        chunks = self.text_splitter.split_documents(documents)

        self.vectorstore.add_documents(chunks)

        return len(chunks)
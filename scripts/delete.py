from ragbot.vectorstore.pinecone_store import PineconeStore

def main():

    store = PineconeStore()

    store.delete_all()

    print("Pinecone cleared")


if __name__ == "__main__":
    main()
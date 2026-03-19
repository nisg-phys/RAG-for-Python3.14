from ragbot.pipeline.rag_pipeline import RAGPipeline

rag = RAGPipeline()

answer = rag.run("What is LangChain?")

print(answer)
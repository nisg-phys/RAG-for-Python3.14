from ragbot.services.ingestion_service import DATA_DIR, run_ingestion
from ragbot.utils.logger import get_logger

logger = get_logger("document_loader")
"""CLI entry point for ingestion. The actual loading/ingestion logic lives in
ragbot.services.ingestion_service, which also backs the POST /ingest endpoint."""


def main():
    result = run_ingestion(DATA_DIR)
    logger.info(
        "Ingestion complete: %s documents loaded, %s chunks created, %s chunks deleted "
        "(%s new sources, %s changed, %s unchanged/skipped, %s removed)",
        result["documents_loaded"],
        result["chunks_created"],
        result["chunks_deleted"],
        result["sources_new"],
        result["sources_changed"],
        result["sources_unchanged"],
        result["sources_removed"],
    )


if __name__ == "__main__":
    main()

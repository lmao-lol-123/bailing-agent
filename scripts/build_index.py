from __future__ import annotations

import argparse
from pathlib import Path

from src.core.dependencies import get_container


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the default Chroma index from local files.")
    parser.add_argument("paths", nargs="+", help="File paths to ingest.")
    parser.add_argument("--force-mineru", action="store_true", help="Force MinerU for PDF files.")
    args = parser.parse_args()

    container = get_container()
    for raw_path in args.paths:
        source_path = Path(raw_path)
        saved_path = container.ingestion_service.copy_local_file(source_path)
        result = container.ingestion_service.ingest_saved_file(saved_path, force_mineru=args.force_mineru)
        print(
            f"Ingested {result.source_name}: docs={result.documents_loaded}, "
            f"chunks={result.chunks_indexed}, mineru={result.used_mineru}"
        )


if __name__ == "__main__":
    main()


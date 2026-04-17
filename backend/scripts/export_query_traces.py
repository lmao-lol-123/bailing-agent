from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.src.core.dependencies import get_container


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export persisted query traces for one chat session."
    )
    parser.add_argument("--session-id", required=True, help="Target chat session id.")
    parser.add_argument("--out", required=True, help="Output JSON file path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max trace records.")
    args = parser.parse_args()

    container = get_container()
    traces = container.chat_history_store.list_query_traces(
        session_id=args.session_id, limit=args.limit
    )
    payload = {
        "session_id": args.session_id,
        "count": len(traces),
        "traces": traces,
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(traces)} traces to {output_path}")


if __name__ == "__main__":
    main()

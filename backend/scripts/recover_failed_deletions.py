from __future__ import annotations

import argparse
import json

from backend.src.core.dependencies import get_container


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry cleanup for session files in recover_pending state."
    )
    parser.add_argument(
        "--session-id", default=None, help="Optional session id. If omitted, scan all sessions."
    )
    args = parser.parse_args()

    container = get_container()
    outcomes = container.ingestion_service.recover_pending_session_file_deletions(
        session_id=args.session_id
    )
    status_counts: dict[str, int] = {}
    for _, _, status in outcomes:
        status_counts[status] = status_counts.get(status, 0) + 1

    if status_counts.get("failed", 0) > 0:
        run_status = "failed"
        exit_code = 2
    elif status_counts.get("skipped", 0) > 0:
        run_status = "partial"
        exit_code = 1
    else:
        run_status = "ok"
        exit_code = 0

    payload = {
        "session_id": args.session_id,
        "run_status": run_status,
        "exit_code": exit_code,
        "count": len(outcomes),
        "status_counts": status_counts,
        "results": [
            {"session_id": session_id, "file_id": file_id, "status": status}
            for session_id, file_id, status in outcomes
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

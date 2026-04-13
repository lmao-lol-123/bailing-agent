# Changelog

## 2026-04-14

### Added

- Added hybrid retrieval building blocks, including lexical recall, deterministic query expansion, query routing, reranking, and targeted retry modules under `backend/src/retrieve/`
- Added index-manager support for active chunk snapshots and parent-content hydration lookup
- Added retrieval-focused tests covering query expansion, query routing, reranking, retry behavior, and vector-store hybrid retrieval
- Added root-level bilingual GitHub-facing documentation in `README.md` and `README.en.md`

### Changed

- Updated the retrieval service to fuse dense and lexical candidates, preserve query-variant metadata, and hydrate parent chunks when child chunks are retrieved
- Updated the parent store to support direct parent-record lookup by persisted reference
- Updated the GitHub Actions workflow so CI runs for both `master` and `main`, in addition to `codex/**` branches
- Updated `docs/README.md` and `docs/README.en.md` to point to the root readmes as the canonical project overview

### Validation

- Ran the full backend test suite: `60 passed`
- Verified the DeepSeek integration with a real streaming smoke test through the backend answer service

### Notes

- Local runtime data under `data/uploads/`, `data/processed/`, and `storage/` remains excluded from Git tracking
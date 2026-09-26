# Ogura

- **Text recognition:** [`textrec/`](textrec/README.md). Run commands after `cd textrec`.
- **Text detection:** [`ogura/textdet/`](ogura/textdet/README.md). Its current environment remains at the repository root.

Textrec has its own `pyproject.toml`, `uv.lock`, scripts, configuration, tests and local data.
Shared fonts, Wikipedia and Unicode source data live in `corpus/` at the repository root.
Recognition datasets, checkpoints and caches live under `textrec/`; no compatibility
symlinks are required. Detection continues using the root corpus paths.

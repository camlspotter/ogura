# Ogura

- **Text recognition:** [`textrec/`](textrec/README.md). Run commands after `cd textrec`.
- **Text detection:** [`ogura/textdet/`](ogura/textdet/README.md). Its current environment remains at the repository root.

Textrec has its own `pyproject.toml`, `uv.lock`, scripts, configuration, tests and local data.
Shared fonts, Wikipedia and Unicode source data live in `corpus/` at the repository root.
Recognition datasets, checkpoints and caches live under `textrec/`; no compatibility
symlinks are required. Detection continues using the root corpus paths.

## One environment for detection and recognition

From the repository root:

```sh
uv sync --frozen --extra textrec
uv run --frozen --extra textrec python -m ogura.textrec.recognize --help
uv run --frozen --extra textrec python -m ogura.textdet.prepare --help
```

`ogura-textdet` and `ogura-textrec` are independent distributions sharing the
PEP 420 `ogura` namespace (there is no `ogura/__init__.py`). Always use the
`ogura.textdet.*` and `ogura.textrec.*` names. Keep `--extra textrec` on root
`uv run` commands so uv retains the optional recognition installation.
Alternatively install both checkouts with `pip install -e . -e ./textrec`.
Development/data commands currently target editable source checkouts; datasets
and models are not bundled in wheels. Relative CLI paths still depend on the
working directory; installation does not switch it automatically.

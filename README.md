# Ogura

- **Text recognition:** [`textrec/`](textrec/README.md). Run commands after `cd textrec`.
- **Text detection:** [`textdet/`](textdet/README.md). Run commands after `cd textdet`.
- **Table structure recognition:** [`tablerec/`](tablerec/README.md). Synthetic table recipes and boundary CNN.

Each package has its own `pyproject.toml`, `uv.lock`, tests and local data.
All three packages belong to this single Git repository.
Shared fonts, Wikipedia and Unicode source data live in `corpus/` at the repository root.
Recognition datasets, checkpoints and caches live under `textrec/`; no compatibility
symlinks are required. Detection data and caches live under `textdet/`. Table recipes and checkpoints live under `tablerec/`.

## One environment for detection, recognition and tables

From the repository root:

```sh
uv sync --frozen --extra textrec
uv run --frozen --extra textrec python -m ogura.textrec.recognize --help
uv run --frozen --extra textrec python -m ogura.textdet.prepare --help
uv run --frozen --extra textrec python -m ogura.tablerec.train_table_cnn --help
```

`ogura-textdet`, `ogura-textrec` and `ogura-tablerec` are independent distributions sharing the
PEP 420 `ogura` namespace (there is no `ogura/__init__.py`). Always use the
`ogura.textdet.*`, `ogura.textrec.*` and `ogura.tablerec.*` names. Keep `--extra textrec` on root
`uv run` commands so uv retains the optional recognition installation.
Alternatively install the checkouts with `pip install -e ./textdet -e ./textrec -e ./tablerec`.
Development/data commands currently target editable source checkouts; datasets
and models are not bundled in wheels. Relative CLI paths still depend on the
working directory; installation does not switch it automatically.

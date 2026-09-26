# laya_rag

A retrieval-augmented generation (RAG) project built on top of
[laya](https://github.com/thread3d/laya).

## Repository layout

| Path | Description |
| --- | --- |
| `laya/` | Submodule pinned to [thread3d/laya](https://github.com/thread3d/laya). |
| `deepseek-harness/` | Submodule pinned to [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness). |

## Getting started

Clone with submodules:

```sh
git clone --recurse-submodules <repository-url> laya_rag
```

If the repository is already cloned, fetch the submodule contents with:

```sh
git submodule update --init --recursive
```

## Working with submodules

Submodules are pinned to a specific commit, not to a branch. After checking
out a new commit inside a submodule, record it in the parent repository:

```sh
git -C laya checkout <ref>
git add laya
git commit -m "Update laya submodule"
```

To move every submodule to its upstream default branch tip:

```sh
git submodule update --remote --merge
```

## License

See the `LICENSE` file in each dependency for its own terms.

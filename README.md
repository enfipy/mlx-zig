# mlx

Unified Zig ML workspace built from:

- `MLX.zig` (Zig bindings + `llm` and `whisper` apps)
- `zig-build-mlx` (Zig-native MLX build graph)
- tigercheck tooling and strict style gates

This README consolidates previous project READMEs into a single source of truth.

## Toolchain

Pinned Zig toolchain: `0.16.0-dev.2637+6a9510c0e`

```bash
./zig/download.sh
./zig/zig version
```

## Build

```bash
./zig/zig build
```

Build installs these executables:

- `mlx` (repo entrypoint)
- `llm`
- `whisper`

Optional path overrides:

```bash
./zig/zig build -Dmodel-assets-dir=src/assets/models -Dfixture-dir=src/assets/fixtures
```

## Run

```bash
./zig/zig build run
./zig/zig build run-llm -- --help
./zig/zig build run-whisper -- src/assets/fixtures/audio/alive.mp3
```

## LLM options

`llm` supports:

- `--config=CONFIG`
- `--format=FORMAT`
- `--model-type=TYPE`
- `--model-name=NAME`
- `--max=N`
- `--temperature=T` (default: `0`, deterministic greedy)
- `--top-p=P` (default: `1.0`)
- `--top-k=K` (default: `0`, disabled)
- `--help`

## Model integrity

- `download_model` now verifies each downloaded model artifact against remote digest headers from Hugging Face before use.
- Supported digests: SHA-256 and SHA-1 (based on remote header metadata).
- Integrity mismatch fails the download flow with an explicit error.

## Quality gates

```bash
./zig/zig build test
./zig/zig build check
./zig/zig build check-strict
```

- `check`: build + tests + tigercheck text diagnostics
- `check-strict`: build + tests + strict tigercheck lane

Override tigercheck executable or target path if needed:

```bash
./zig/zig build check -Dtigercheck-exe=../tigercheck/zig-out/bin/tigercheck -Dstyle-path=src/libmlx
```

## Build internals

- `install-mlx-c`: builds `mlx-c` and MLX static libraries through Zig build graph dependencies
- `build-mlx-core`: builds core MLX static library via `src/build/zig-build-mlx.zig`

## Repository layout

- `src/libmlx/` bindings, model layers, and runtime utilities
- `src/tools/` CLI apps (`llm`, `whisper`)
- `src/mlx/` default repo entrypoint
- `src/build/` zig-build-mlx integration files
- `docs/` diagrams and project docs assets
- `src/assets/models/` tokenizer/model assets
- `src/assets/fixtures/audio/` Whisper audio fixtures
- `src/assets/fixtures/manifest.json` fixture checksums and byte sizes

Tokenizer round-trip test path override (optional):

```bash
MLX_MODEL_ASSETS_DIR=src/assets/models ./zig/zig build test
```

## Platform notes

- Primary target: Apple Silicon macOS
- MLX backend build is wired through Zig dependencies (no top-level CMake shell-out)

## License

Apache License 2.0 (`LICENSE`)

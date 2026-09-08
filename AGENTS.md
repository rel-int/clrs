# AGENTS.md

## What

DeepMind's [CLRS Algorithmic Reasoning Benchmark](README.md), forked to `rel-int/clrs` as the
beachhead for [discopy#678](https://github.com/discopy/discopy/issues/678): CLRS tasks run as
string diagrams whose wiring *is* the algorithm, evaluated by a DisCoPy functor with JAX arrays on
the wires, and only the boxes learned. The experiment and its results are [`goi/`](goi/README.md).
It moved here from `toumix/clrs` on 2026-09-08, history preserved (its pull requests #1 and #2).

## Where

- [`goi/`](goi/) — the experiment: `adapter.py` samples tasks through `clrs._src.samplers` with
  `stubs.py` standing in for TensorFlow, one module per task (`minimum`, `sort`, `lcs`), the
  runners `python -m goi.run_<task>`, and [`lake.py`](goi/lake.py), the one writer and reader of
  runs on the data lake. [`goi/README.md`](goi/README.md) carries every number with its protocol.
- `clrs/` — upstream, untouched. Nothing of the experiment goes there.

## Infra: one contract, read it before touching data or compute

Credentials, the data lake and remote compute are documented once, in
[`rel-int/infra` → `agents/RELINT-AGENTS.md`](https://github.com/rel-int/infra/blob/main/agents/RELINT-AGENTS.md).
Read it first; do not copy it here. From a sandbox:

```sh
cat ../infra/agents/RELINT-AGENTS.md 2>/dev/null || curl -sSf \
  -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.raw" \
  https://api.github.com/repos/rel-int/infra/contents/agents/RELINT-AGENTS.md
```

If neither works, stop and say so. Never guess a credential, a bucket or a path.

A provisioned sandbox holds `RELINT_TOKEN` in its environment and nothing else. The one line
this repo needs, run once per session (it is also the refresh):

```sh
curl -H "Authorization: Bearer $RELINT_TOKEN" https://data.rel-int.ai/setup.py | uv run - --scratch
source ~/.config/relint/env
```

What the contract does not say and this repo measured (2026-09-08, from a Claude Code sandbox):

- A scratch key starts with `$RELINT_SCRATCH_PREFIX`, the caller's e-mail with every character
  outside `[a-zA-Z0-9._-]` replaced by `_` (`scratch/alexis_rel-int.ai/`), which the doctor exports
  only with `--scratch`. Any other prefix is `AccessDenied`, indistinguishable from a real refusal.
- `--append` credentials are rejected by R2 on every request, reads included
  (`InvalidArgument: X-Amz-Security-Token`), so landing under `raw/` is not possible today. Use
  scratch, or `--write` if your row in `users.yaml` carries `data:write`.
- The doctor overwrites `~/.config/relint/env` with the last credential requested: one shell,
  one bucket. `goi.lake` reads `RELINT_BUCKET` to know which it has.

## Running experiments

Every table in [`goi/README.md`](goi/README.md) is minutes of CPU, so the default is the sandbox
itself, nothing remote:

```sh
uv venv --python 3.12 .venv && . .venv/bin/activate    # discopy main needs 3.12
uv pip install numpy jax attrs absl-py chex dm-haiku optax ml_collections \
    six opt-einsum matplotlib pytest boto3 "git+https://github.com/discopy/discopy@main"
python -m pytest goi/minimum_test.py goi/sort_test.py goi/lake_test.py
python -u -m goi.run_minimum          # -u: a buffered run looks dead for its whole duration
```

Every runner ends by saving each seed's scorecard and parameters with `goi.lake`, to
`$RELINT_SCRATCH_PREFIX/clrs/<git sha>/<task>/seed-<k>/` when the shell carries a credential and
to nowhere, saying so, when it does not. Scratch is deleted after thirty days; a result worth
keeping is landed by hand at `weights/clrs/<task>/<date>/seed-<k>/` with a `--write` credential,
and the date is the version key, as `rel-int/infra`'s `datasets.toml` keys everything.

For a run that needs a GPU or more wall-clock than a sandbox gives: marimohub
(`MARIMOHUB_TOKEN`, delivered by the exchange above) runs notebooks, and Modal runs scripts —
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` in the environment and `modal[api-proxy-support]` installed,
as `toumix/ARC-AGI`'s launcher does. Neither has been needed here yet.

## Never

- Never commit a dataset, a checkpoint, a parameter file or a run's output to git. The numbers go
  in `goi/README.md`; the artefacts go to the lake or nowhere.
- Never invent a bucket, a prefix or a dataset path: `rel-int/infra`'s `datasets.toml` is the
  index of what exists and where.
- Never mint, paste or store a credential other than `RELINT_TOKEN` in the environment; never
  widen a scope or touch `rel-int/infra`'s credential stack to make a run pass.

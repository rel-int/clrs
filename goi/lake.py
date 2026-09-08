"""Save and load `goi` runs on the rel-int data lake.

A run is a task and a seed; its artefacts are the scorecard -- one score
per split -- and the trained parameters, a flat dict of arrays. The lake
keeps them as two objects under one prefix per run:

    <run prefix>/scores.json
    <run prefix>/params.npz

While a run is in flight the prefix is the caller's own scratch space,
`$RELINT_SCRATCH_PREFIX/clrs/<git sha>/<task>/seed-<k>/`, deleted after
thirty days; a landed run lives at `weights/clrs/<task>/<date>/seed-<k>/`
on the data bucket, immutable, written with a curator credential. The
version key of a landed run is a date, as `rel-int/infra`'s
`datasets.toml` keys every dataset and weight, and the git sha of the
code that produced it is recorded inside `scores.json` instead.

Credentials are read from the environment and nowhere else, exactly as
`rel-int/infra`'s `agents/RELINT-AGENTS.md` delivers them: the doctor
writes `~/.config/relint/env`, `source` exports it, and `RELINT_BUCKET`
says which bucket the credential is for.
"""

import datetime
import io
import json
import os
import subprocess

import numpy as np

SETUP = ('curl -H "Authorization: Bearer $RELINT_TOKEN" '
         'https://data.rel-int.ai/setup.py | uv run - --scratch '
         '&& source ~/.config/relint/env')


def configured():
  """Whether the environment carries a lake credential at all."""
  return all(name in os.environ
             for name in ('AWS_ACCESS_KEY_ID', 'AWS_ENDPOINT_URL_S3',
                          'RELINT_BUCKET'))


def git_sha():
  """The sha of the code producing the run, or `unknown` outside git."""
  try:
    return subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
        text=True, check=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return 'unknown'


def prefix(task, seed, version=None):
  """The run's key prefix: scratch in flight, `weights/` once landed."""
  scratch = os.environ.get('RELINT_SCRATCH_PREFIX')
  if scratch:
    return f"{scratch}clrs/{version or git_sha()}/{task}/seed-{seed}/"
  version = version or datetime.date.today().isoformat()
  return f"weights/clrs/{task}/{version}/seed-{seed}/"


def client():
  """An S3 client on the lake, from the variables the doctor exported."""
  import boto3  # pylint: disable=g-import-not-at-top
  return boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
                      region_name='auto')


def save(task, seed, scores, params, s3=None, version=None):
  """Write one run's scorecard and parameters; returns the keys written."""
  s3 = s3 or client()
  bucket, run = os.environ['RELINT_BUCKET'], prefix(task, seed, version)
  record = {'task': task, 'seed': seed, 'sha': git_sha(),
            'created': datetime.datetime.now(datetime.UTC).isoformat(),
            'scores': scores}
  arrays = io.BytesIO()
  np.savez(arrays, **{key: np.asarray(value) for key, value in params.items()})
  s3.put_object(Bucket=bucket, Key=run + 'scores.json',
                Body=json.dumps(record, default=float).encode())
  s3.put_object(Bucket=bucket, Key=run + 'params.npz', Body=arrays.getvalue())
  return [run + 'scores.json', run + 'params.npz']


def load(task, seed, version, s3=None):
  """Read one run back: its record and its parameters as a flat dict."""
  s3 = s3 or client()
  bucket, run = os.environ['RELINT_BUCKET'], prefix(task, seed, version)
  record = json.load(s3.get_object(Bucket=bucket, Key=run + 'scores.json')['Body'])
  with np.load(io.BytesIO(
      s3.get_object(Bucket=bucket, Key=run + 'params.npz')['Body'].read())) as f:
    params = {key: f[key] for key in f.files}
  return record, params


def save_runs(task, seeds, results, s3=None):
  """Save every seed's `(scores, params)` when a credential is present.

  Says where each run went, or that it stayed in the shell and how to
  change that; a run is never dropped silently.
  """
  if not configured():
    print(f"\nno lake credential in the environment, so the {task} runs "
          f"stay in this shell; to save them:\n  {SETUP}")
    return []
  keys = [key for seed, (scores, params) in zip(seeds, results)
          for key in save(task, seed, scores, params, s3)]
  for key in keys:
    print(f"saved r2://{os.environ['RELINT_BUCKET']}/{key}")
  return keys

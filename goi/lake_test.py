"""Unit tests for `goi.lake`, against an in-memory bucket."""

import io
import os
from unittest import mock

import numpy as np

from absl.testing import absltest

from goi import lake


class Bucket:
  """The two S3 calls `goi.lake` makes, over a dict."""

  def __init__(self):
    self.objects = {}

  def put_object(self, Bucket, Key, Body):  # pylint: disable=invalid-name
    self.objects[Bucket, Key] = Body

  def get_object(self, Bucket, Key):  # pylint: disable=invalid-name
    return {'Body': io.BytesIO(self.objects[Bucket, Key])}


SCRATCH = {'AWS_ACCESS_KEY_ID': 'k', 'AWS_SECRET_ACCESS_KEY': 's',
           'AWS_ENDPOINT_URL_S3': 'https://r2',
           'RELINT_BUCKET': 'rel-int-scratch',
           'RELINT_SCRATCH_PREFIX': 'scratch/a_b.ai/'}
CURATOR = {'AWS_ACCESS_KEY_ID': 'k', 'AWS_SECRET_ACCESS_KEY': 's',
           'AWS_ENDPOINT_URL_S3': 'https://r2',
           'RELINT_BUCKET': 'rel-int-data'}


class LakeTest(absltest.TestCase):

  def test_scratch_prefix_is_the_signed_one(self):
    with mock.patch.dict(os.environ, SCRATCH, clear=True):
      self.assertEqual(lake.prefix('minimum', 0, version='abc123'),
                       'scratch/a_b.ai/clrs/abc123/minimum/seed-0/')

  def test_landed_prefix_is_keyed_by_date(self):
    with mock.patch.dict(os.environ, CURATOR, clear=True):
      self.assertEqual(lake.prefix('minimum', 2, version='2026-09-08'),
                       'weights/clrs/minimum/2026-09-08/seed-2/')

  def test_round_trip(self):
    bucket = Bucket()
    scores = {'test (n=64)': (np.float32(0.97), 0.85)}
    params = {'w1': np.ones((2, 3)), 'b1': np.zeros(3)}
    with mock.patch.dict(os.environ, SCRATCH, clear=True):
      keys = lake.save('minimum', 1, scores, params, s3=bucket, version='v')
      record, loaded = lake.load('minimum', 1, 'v', s3=bucket)
    self.assertEqual(keys, ['scratch/a_b.ai/clrs/v/minimum/seed-1/scores.json',
                            'scratch/a_b.ai/clrs/v/minimum/seed-1/params.npz'])
    self.assertEqual(record['task'], 'minimum')
    self.assertAlmostEqual(record['scores']['test (n=64)'][0], 0.97, places=5)
    np.testing.assert_array_equal(loaded['w1'], params['w1'])
    np.testing.assert_array_equal(loaded['b1'], params['b1'])

  def test_save_runs_configured(self):
    bucket = Bucket()
    results = [({'val (n=16)': 1.0}, {'w': np.zeros(2)}),
               ({'val (n=16)': 0.5}, {'w': np.ones(2)})]
    with mock.patch.dict(os.environ, SCRATCH, clear=True):
      with mock.patch('builtins.print') as fake_print:
        keys = lake.save_runs('minimum', (0, 1), results, s3=bucket)
      self.assertLen(keys, 4)
      self.assertLen(bucket.objects, 4)
      record, params = lake.load('minimum', 1, lake.git_sha(), s3=bucket)
    self.assertEqual(record['scores'], {'val (n=16)': 0.5})
    np.testing.assert_array_equal(params['w'], np.ones(2))
    self.assertEqual(fake_print.call_count, 4)
    self.assertIn('r2://rel-int-scratch/', fake_print.call_args[0][0])

  def test_unconfigured_saves_nothing_and_says_so(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch('builtins.print') as fake_print:
        self.assertEqual(lake.save_runs('minimum', (0,), [({}, {})]), [])
    self.assertIn('setup.py', fake_print.call_args[0][0])

  def test_partial_credential_is_not_configured(self):
    partial = {key: value for key, value in SCRATCH.items()
               if key != 'AWS_SECRET_ACCESS_KEY'}
    with mock.patch.dict(os.environ, partial, clear=True):
      self.assertFalse(lake.configured())


if __name__ == '__main__':
  absltest.main()

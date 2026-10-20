# TODO

> let's close this PR and move its wip to the clrs repo if it's not there already https://github.com/discopy/discopy/pull/686

- [x] Check whether the GoNI work of discopy#686 is already here: it is not — `goi/` (rel-int/clrs#1) carries the earlier `minimum`/`sort`/hand-wired `lcs` runs, not the `kmp_matcher` study or the LCS grid as a diagram
- [x] Move `docs/neural/examples/GoNI/` and `test/neural/test_goni.py` of discopy#686 at `359e80f` to `goni/`, without the `.npz`/`.pt` data and weights `AGENTS.md` keeps out of git
- [x] Close discopy#686 with a pointer here
- [ ] Port `goni/` from the `discopy.neural` of discopy#585's host branch to the one on `rel-int/discopy-next` (`CMap.permutation` landed there already as #735)
- [ ] Regenerate the splits with `dataset.py --generate` / `lcs_dataset.py --generate`, re-run the three seeds of each study, save weights through `goi.lake` rather than `artifacts/`
- [ ] Re-check every number of `goni/README.md` against those runs

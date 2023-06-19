# Developer guide

## Developer setup
An [editable installation](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
is preferable for developers:
```sh
# from your local clone:
pip install -e .
```

## Continuous deployment
The continuous deployment (CD) pipeline uses
[`python-semantic-release`](https://github.com/relekang/python-semantic-release) and
[`setuptools_scm`](https://github.com/pypa/setuptools_scm) to automatically manage
versioning and releasing. For each commit to `main`, i.e. upon merging, the pipeline
assesses if a release is needed and –if that is the case– creates and pushes a Git tag, a
GitHub release, and a PyPI/Artifactory package version.

To check if a release is required and determine the target version,
python-semantic-release parses the relevant commit messages assuming
[Angular style](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#commits),
and decides *if* and *how* to bump the version
(as per [SemVer](https://semver.org)) based on config options `parser_angular_patch_types`
and `parser_angular_minor_types`.

Consider an example where the current version is `3.4.1` and `pyproject.toml` includes:
```
# ...
[tool.semantic_release]
parser_angular_allowed_types = "build,chore,ci,docs,feat,fix,perf,style,refactor,test"
parser_angular_patch_types = "fix,perf"
parser_angular_minor_types = "feat"
# ...
```

Then:
- merging commit `"docs: update README"` would *not* lead to a new release
- merging commit `"perf: optimize parser"` would lead to new release `3.4.2` (patch bump)
- merging commit `"feat: add parser"` would lead to a new release `3.5.0` (minor bump)
- merging a commit with a `"BREAKING CHANGE:"`
  [footer](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#footer) would
  lead to a new release `4.0.0` (major bump)
    - note that this logic is currently hard-coded in python-semantic-release, i.e. it is
      currently not possible to configure custom cues/patterns for triggering major bumps.

Note that each type defined in `parser_angular_patch_types` or `parser_angular_minor_types`
should also be in `parser_angular_allowed_types`, otherwise it will not trigger.

For more details about python-semantic-release check its
[docs](https://python-semantic-release.readthedocs.io) and [default config](https://github.com/relekang/python-semantic-release/blob/master/semantic_release/defaults.cfg),
as well as MMMT's local config in `pyproject.toml`, under `[tool.semantic_release]`.

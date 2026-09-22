# Contributing to AdsDrift

AdsDrift is released as a research prototype whose original development is
currently paused. New maintainers, reproductions, bug fixes, documentation
improvements, and extensions are welcome.

## Good places to start

- Reproduce the existing generator and loss tests in a documented environment.
- Replace machine-specific training paths with portable configuration.
- Add a small, redistributable example dataset and an end-to-end smoke test.
- Package the project for a standard `pip install -e .` workflow.
- Validate generated structures with independent energy and force calculations.
- Improve inference so it does not require a complete training feature bank.

Before making a large change, open an issue describing the proposed scope and
validation plan. Pull requests should state what changed, how it was tested,
and any scientific assumptions or evidence boundaries affected by the change.

By contributing, you agree that your contribution is licensed under the
Apache License 2.0 used by this repository.

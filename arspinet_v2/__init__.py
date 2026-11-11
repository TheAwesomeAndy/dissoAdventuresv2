"""ARSPI-Net v2: the upgraded measurement chain.

v2 keeps the v1 confirmatory protocol (participant-grouped CV, fold-local fitting,
fixed data-independent SRP-64) and changes what the external methods review of v1
identified as unsound or untested:

* ``reservoir``   one batched LIF implementation that reproduces the v1 operator and
                  exposes the recurrent gain, reservoir sharing and a non-spiking control.
* ``dynamics``    fading-memory tests run on the reservoir that is actually deployed.
* ``features``    matched non-reservoir controls and windowing in true milliseconds.
* ``evaluation``  the headline estimator, fast participant bootstrap, and a permutation
                  test that uses the headline estimator instead of a ridge stand-in.
* ``controls``    destruction controls that separate information removal from noise
                  injection (train-and-test versus test-only).
* ``nuisance``    calibrated subject and condition variance components with nulls.

See V2_UPGRADE.md at the repository root for the finding-by-finding record.
"""

__version__ = "2.0.0"

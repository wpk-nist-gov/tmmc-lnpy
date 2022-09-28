"""
define optional paremeters
"""

import os

TQDM_USE = "tqdm_use"
TQDM_LEN_CALC = "tqdm_len_calc"
TQDM_LEN_BUILD = "tqdm_len_build"
TQDM_LEAVE = "tqdm_leave"
TQDM_BAR = "tqdm_bar"

JOBLIB_USE = "joblib_use"
JOBLIB_N_JOBS = "joblib_n_jobs"
JOBLIB_BACKEND = "joblib_backend"
JOBLIB_KWS = "joblib_kws"
JOBLIB_LEN_CALC = "joblib_len_calc"
JOBLIB_LEN_BUILD = "joblib_len_build"

DOC_SUB = "doc_sub"

try:
    # Default to setting docs
    _DOC_SUB = os.getenv("LNPI_DOC_SUB", "True").lower() not in ("0", "f", "false")
except KeyError:
    _DOC_SUB = True


OPTIONS = {
    TQDM_USE: True,
    TQDM_LEN_CALC: 100,
    TQDM_LEN_BUILD: 500,
    TQDM_LEAVE: False,
    TQDM_BAR: "default",
    JOBLIB_USE: True,
    JOBLIB_N_JOBS: -1,
    JOBLIB_BACKEND: None,
    JOBLIB_KWS: {},
    JOBLIB_LEN_CALC: 200,
    JOBLIB_LEN_BUILD: 500,
    DOC_SUB: _DOC_SUB,
}

_isbool = lambda x: isinstance(x, bool)
_isint = lambda x: isinstance(x, int)
_isstr = lambda x: isinstance(x, str)

_isstr_or_None = lambda x: isinstance(x, str) or x is None

_VALIDATORS = {
    TQDM_USE: _isbool,
    TQDM_LEN_CALC: _isint,
    TQDM_LEN_BUILD: _isint,
    TQDM_LEAVE: _isbool,
    TQDM_BAR: lambda x: x in ["default", "text", "notebook"],
    JOBLIB_USE: _isbool,
    JOBLIB_N_JOBS: _isint,
    JOBLIB_BACKEND: _isstr_or_None,
    JOBLIB_KWS: lambda x: isinstance(x, dict),
    JOBLIB_LEN_CALC: _isint,
    JOBLIB_LEN_BUILD: _isint,
    DOC_SUB: _isbool,
}

_SETTERS = {}


class set_options(object):
    """Set options for xarray in a controlled context.
    Currently supported options:
    - `tqdm_use` : if `True`, use progress bar where appropriate
    - `tqdm_len_calc` : min length for using bar in calculations of properties
    - `tqdm_len_build` : min length for building Collection objects
    - `tqdm_leave` : if True, leave bar after execution.  Default=False
    - `joblib_use` : if `True`, use joblib
    - `joblib_n_jobs` : number of processors to use, default=all processors
    - `joblib_backend` : backend to use.  Default='multiprocessing'.
    - `joblib_kws` : dictionary of arguments to joblib.Parallel.
    - `joblib_len_calc` : min length to use parallel in calculations
    - `joblib_len_build` : min lenght to use parallel in build
    You can use ``set_options`` either as a context manager:
    >>> with xr.set_options(use_tqdm=True, tqdm_min_len_calc=50):
    ...     c.xge.betaOmega()
    ...
    Or to set global options:
    >>> xr.set_options(tqdm_min_len_calc=50)
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name %r is not in the set of valid options %r"
                    % (k, set(OPTIONS))
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        for k, v in options_dict.items():
            if k in _SETTERS:
                _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)

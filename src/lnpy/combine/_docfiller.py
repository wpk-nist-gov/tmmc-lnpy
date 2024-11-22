from __future__ import annotations

from module_utilities.docfiller import DocFiller

from lnpy.core.docstrings import docfiller

_docstrings_local = r"""
Parameters
----------
lnpi_name : str
    Column/variable name corresponding to :math:`\ln \Pi(N)`.
delta_lnpi_name : str
    Column/variable name corresponding to :math:`\Delta \ln \Pi(N)`.
window_name :
    Column name corresponding to "window", i.e., an individual simulation.
    Note that this is only used if passing in a single dataframe with multiple windows.
state_name :
    Column name corresponding to simulation state. For example, ``state="state"``.
macrostate_names :
    Column name(s) corresponding to a single "state". For example, for a single
    component system, this could be ``macrostate_names="n"``, and for a binary
    system ``macrostate_names=["n_0", "n_1"]``
up_name :
    Column name corresponding to "up" probability.
down_name :
    Column name corresponding to "down" probability.
weight_name :
    Column name corresponding to "weight" of probability.
table_assign | table :
    :class:`pandas.DataFrame` or :class:`xarray.Dataset` data container.
check_connected :
    If ``True``, check that all windows form a connected graph.
tables :
    Individual sample windows. If pass in a single
    :class:`~pandas.DataFrame`, it must contain the column ``window_name``.
    Otherwise, the individual frames will be concatenated and the
    ``window_name`` column will be added (or replaced if already present).
up :
    Probability of moving from ``state[i]`` to ``state[i+1]``.
down :
    Probability of moving from ``state[i]`` to ``state[i-1]``.
norm :
    If True, normalize :math:`\ln \Pi(N)`.
normalize : bool
    If ``True``, normalize probability.
array_name | name:
    Optional name to assign to the output :class:`pandas.Series` or `xarray.DataArray`.

index : array-like, optional
    Index into `axis` of `data`.  Defaults to range(len(down))
groups | group_start, group_end : array-like, optional
    Start, end of index for a group.
    ``index[group_start[group]:group_end[group]]`` are the indices for
    group ``group``.  Defaults to single group

axis : int, optional
    Axis to calculate along.
dim : str, optional
    Dimension to calculate along.  Overrides `axis` if specified.
out : ndarray, optional
    Output array.
dtype : dtype, optional
    Optional :class:`~numpy.dtype` for output data.


Raises
------
OverlapError
    If the overlaps do not form a connected graph, then raise a ``OverlapError``.

"""


docfiller_local = docfiller.append(
    DocFiller.from_docstring(_docstrings_local, combine_keys="parameters")
).decorate

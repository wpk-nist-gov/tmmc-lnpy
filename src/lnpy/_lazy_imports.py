from typing import TYPE_CHECKING

# Have this to support type checking and IDE support
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xarray as xr
else:
    import lazy_loader as lazy

    np = lazy.load("numpy")
    pd = lazy.load("pandas")
    xr = lazy.load("xarray")

__all__ = ["np", "pd", "xr"]

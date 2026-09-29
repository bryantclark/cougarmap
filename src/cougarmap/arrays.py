"""Array type aliases shared by the model. Rasters are 2-D (row 0 at the north edge) unless a name says otherwise."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

type Floats = npt.NDArray[np.floating[Any]]  # a float raster (0-1 layers, metres, seconds, degrees)
type Mask = npt.NDArray[np.bool_]  # a yes/no raster
type Ints = npt.NDArray[np.integer[Any]]  # labels, counts, class codes, flat cell indices

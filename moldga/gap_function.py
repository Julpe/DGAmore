import numpy as np

from moldga.brillouin_zone import KGrid
from moldga.local_n_point import LocalNPoint
from moldga.n_point_base import IAmNonLocal, IHaveChannel, SpinChannel, FrequencyNotation


class GapFunction(IAmNonLocal, LocalNPoint, IHaveChannel):
    """
    Represents the superconducting gap function. Has one momentum dimension, two orbital dimensions and one fermionic
    frequency dimension.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        nk: tuple[int, int, int] = (1, 1, 1),
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
    ):
        LocalNPoint.__init__(self, mat, 2, 0, 1, full_niv_range=full_niv_range)
        IAmNonLocal.__init__(self, mat, nk, has_compressed_q_dimension=has_compressed_q_dimension)
        IHaveChannel.__init__(self, channel, FrequencyNotation.PP)

    def map_to_full_bz(self, k_grid: KGrid, nq: tuple = None):
        return self._map_to_full_bz(k_grid, 2, nq)

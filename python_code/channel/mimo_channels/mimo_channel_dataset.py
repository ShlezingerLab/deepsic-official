from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import BPSKModulator
from python_code.utils.constants import N_USERS, N_ANTS


class MIMOChannel:
    def __init__(self, block_length: int, pilots_length: int, fading_in_channel: bool):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = N_USERS
        self._h_shape = [N_ANTS, N_USERS]
        self.rx_length = N_ANTS
        self.fading_in_channel = fading_in_channel

    def _transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilots_length, N_USERS))
        tx_data = self._bits_generator.integers(0, 2, size=(self._block_length - self._pilots_length, N_USERS))
        tx = np.concatenate([tx_pilots, tx_data])
        # modulation
        s = BPSKModulator.modulate(tx.T)
        # pass through channel
        rx = SEDChannel.transmit(s=s, h=h, snr=snr)
        return tx, rx.T

    def _transmit_and_detect(self, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        h = SEDChannel.calculate_channel(N_ANTS, N_USERS, index, self.fading_in_channel)
        tx, rx = self._transmit(h, snr)
        return tx, h, rx

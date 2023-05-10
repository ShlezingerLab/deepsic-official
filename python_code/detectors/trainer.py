import random
from typing import List, Tuple

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from python_code import DEVICE, conf
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.metrics import calculate_ber

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Trainer(object):
    """
    Implements the meta-trainer class. Every trainer must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited trainer must implement.
    """

    def __init__(self):
        # initialize matrices, dataset and detector
        self.lr = 5e-3
        self.is_online_training = True
        self._initialize_dataloader()
        self._initialize_detector()

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self):
        """
        Every trainer must have some base detector
        """
        self.detector = None

    # calculate train loss
    def _calculate_loss(self, est: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def _deep_learning_setup(self,single_model):
        """
        Sets up the optimizer and loss criterion
        """
        self.optimizer = Adam(filter(lambda p: p.requires_grad, single_model.parameters()), lr=self.lr)
        self.criterion = CrossEntropyLoss().to(DEVICE)

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                                   pilots_length=conf.pilot_size,
                                                   blocks_num=conf.blocks_num,
                                                   fading_in_channel=conf.fading_in_channel)

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Every detector trainer must have some function to adapt it online
        """
        pass

    def _forward(self, rx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Every trainer must have some forward pass for its detector
        """
        pass

    def evaluate(self) -> List[float]:
        """
        The online evaluation run. Main function for running the experiments of sequential transmission of pilots and
        data blocks for the paper.
        :return: list of ber per timestep
        """
        total_ber = []
        # draw the test words for a given snr
        transmitted_words, received_words, hs = self.channel_dataset.__getitem__(snr_list=[conf.snr])
        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            # get current word and channel
            tx, h, rx = transmitted_words[block_ind], hs[block_ind], received_words[block_ind]
            # split words into data and pilot part
            tx_pilot, tx_data = tx[:conf.pilot_size], tx[conf.pilot_size:]
            rx_pilot, rx_data = rx[:conf.pilot_size], rx[conf.pilot_size:]
            # online training main function
            if self.is_online_training:
                self._online_training(tx_pilot, rx_pilot)
            # detect data part after training on the pilot part
            detected_word = self._forward(rx_data)
            # calculate accuracy
            target = tx_data[:, :rx.shape[1]]
            ber = calculate_ber(detected_word, target)
            total_ber.append(ber)
            print(f'current: {block_ind, ber}')
        print(f'Final BER: {sum(total_ber) / len(total_ber)}')
        return total_ber

    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self._calculate_loss(est=est, tx=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss

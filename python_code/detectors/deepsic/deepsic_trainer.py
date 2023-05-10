from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.deepsic.deepsic_detector import DeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, N_USERS, N_ANTS
from python_code.utils.probs_utils import prob_to_BPSK_symbol

ITERATIONS = 5
EPOCHS = 300

Softmax = torch.nn.Softmax(dim=1)

class DeepSICTrainer(Trainer):

    def __init__(self):
        self.lr = 5e-3
        super().__init__()

    def __str__(self):
        return 'DeepSIC'

    def _initialize_detector(self):
        self.detector = [[DeepSICDetector().to(DEVICE) for _ in range(ITERATIONS)] for _ in
                         range(N_USERS)]  # 2D list for Storing the DeepSIC Networks

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        single_model = single_model.to(DEVICE)
        self._deep_learning_setup(single_model)
        y_total = self._preprocess(rx)
        loss = 0
        for _ in range(EPOCHS):
            soft_estimation = single_model(y_total)
            current_loss = self.run_train_loop(soft_estimation, tx)
            loss += current_loss

    def _train_models(self, model: List[List[DeepSICDetector]], i: int, tx_all: List[torch.Tensor],
                      rx_all: List[torch.Tensor]):
        for user in range(N_USERS):
            self._train_model(model[user][i], tx_all[user], rx_all[user])

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main training function for DeepSIC trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        initial_probs = self._initialize_probs(tx)
        tx_all, rx_all = self._prepare_data_for_training(tx, rx, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self._train_models(self.detector, 0, tx_all, rx_all)
        # Initializing the probabilities
        probs_vec = self._initialize_probs_for_training(tx)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self._calculate_posteriors(self.detector, i, probs_vec, rx)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self._prepare_data_for_training(tx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self._train_models(self.detector, i, tx_all, rx_all)

    def _calculate_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=est, target=tx.long())

    @staticmethod
    def _preprocess(rx: torch.Tensor) -> torch.Tensor:
        return rx.float()

    def _forward(self, rx: torch.Tensor) -> torch.Tensor:
        # detect and decode
        probs_vec = self._initialize_probs_for_infer()
        for i in range(ITERATIONS):
            probs_vec = self._calculate_posteriors(self.detector, i + 1, probs_vec, rx)
        return self._compute_output(probs_vec)

    def _compute_output(self, probs_vec):
        symbols_word = prob_to_BPSK_symbol(probs_vec.float())
        detected_word = BPSKModulator.demodulate(symbols_word)
        return detected_word

    def _prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        for k in range(N_USERS):
            idx = [user_i for user_i in range(N_USERS) if user_i != k]
            current_y_train = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            tx_all.append(tx[:, k])
            rx_all.append(current_y_train)
        return tx_all, rx_all

    def _initialize_probs_for_training(self, tx):
        return HALF * torch.ones(tx.shape).to(DEVICE)

    def _initialize_probs(self, tx):
        return tx.clone()

    def _calculate_posteriors(self, model: List[List[nn.Module]], i: int, probs_vec: torch.Tensor,
                              rx: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(N_USERS):
            idx = [user_i for user_i in range(N_USERS) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            preprocessed_input = self._preprocess(input)
            with torch.no_grad():
                output = Softmax(model[user][i - 1](preprocessed_input))
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec

    def _initialize_probs_for_infer(self):
        return HALF * torch.ones(conf.block_length - conf.pilot_size, N_ANTS).to(DEVICE).float()

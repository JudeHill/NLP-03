import logging
from typing import Callable, List

import torch
from datasets import Dataset
from tqdm import tqdm
import pos_tagging.gpu_check as gpu_check

from pos_tagging.base import BaseUnsupervisedClassifier

logger = logging.getLogger()


class HMMClassifier(BaseUnsupervisedClassifier):
    def __init__(self, num_states, num_obs, device=None):
        """
        For N hidden states and M observations,
            transition_prob: (N+1) * (N+1), with [0, :] as initial probabilities
            emission_prob: N * M

        Parameters:
            num_states: number of hidden states
            num_obs: number of observations
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device {self.device}")
        gpu_check.check()

        self.num_states = num_states   # S
        self.num_obs = num_obs         # V
        self.epsilon = 1e-12
        self._FNEG = -1e9  # large negative for masking in log-space

        # Parameters start as random positive values (counts-ish)
        A = torch.rand(self.num_states + 1, self.num_states + 1, device=self.device)
        A[:, 0] = 0.0  # no transitions *to* dummy start
        B = torch.rand(self.num_states, self.num_obs, device=self.device)

        self.transition_prob = A
        self.emission_prob = B
        self.log_scale = False
        self.logify()

        # step counter for sEM
        if not hasattr(self, "cnt"):
            self.cnt = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _batchify_sentences(self, inputs: Dataset, batch_size: int):
        """
        Yield batches of (obs, lengths) from a HuggingFace Dataset.

        obs:      (B, T_max) LongTensor
        lengths:  (B,) LongTensor
        """
        N = len(inputs)
        for start in range(0, N, batch_size):
            batch = inputs[start:start + batch_size]
            # `batch` is a list of dicts
            lens = [len(eg["input_ids"]) for eg in batch]
            T_max = max(lens)
            B = len(batch)

            obs = torch.zeros(B, T_max, dtype=torch.long, device=self.device)
            lengths = torch.tensor(lens, dtype=torch.long, device=self.device)

            for i, eg in enumerate(batch):
                ids = eg["input_ids"]
                L = len(ids)
                obs[i, :L] = torch.tensor(ids, dtype=torch.long, device=self.device)

            yield obs, lengths

    @staticmethod
    def _normalize(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Row-normalise matrix to sum to 1 (in-place, linear space).
        """
        row_sums = mat.sum(dim=-1, keepdim=True)
        mask = row_sums > 0
        mat[mask] = mat[mask] / (row_sums[mask] + eps)
        return mat

    def _normalize_log_counts(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Given *non-negative counts* in `mat`, convert in-place to log-probabilities:
        log p_ij = log(count_ij + eps) - log(sum_j(count_ij + eps)).
        """
        log_counts = torch.log(mat + self.epsilon)
        log_row_sums = torch.logsumexp(log_counts, dim=-1, keepdim=True)
        mat.copy_(log_counts - log_row_sums)
        return mat

    def logify(self):
        """
        Convert self.transition_prob and self.emission_prob (counts/weights)
        into row-normalised log-probabilities (in-place).
        """
        self._normalize_log_counts(self.transition_prob)
        self._normalize_log_counts(self.emission_prob)
        self.log_scale = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self):
        A = torch.rand(self.num_states + 1, self.num_states + 1, device=self.device)
        A[:, 0] = 0.0
        B = torch.rand(self.num_states, self.num_obs, device=self.device)
        self.transition_prob = A
        self.emission_prob = B
        self.log_scale = False
        self.logify()

    def train(
        self,
        inputs: Dataset,
        epochs: int = 5,
        method: str = "mle",
        continue_training=False,
        batch_size: int = 32,
    ) -> None:
        if method == "mle":
            self.train_logmle(inputs)
        elif method == "EM":
            self.train_EM_log(
                inputs,
                num_iter=epochs,
                continue_training=continue_training,
                batch_size=batch_size,
            )
        elif method == "sEM":
            self.train_sEM(
                inputs,
                num_iter=epochs,
                eta_fn=lambda k: (k + 2) ** (-1.0),
                continue_training=continue_training,
                batch_size=batch_size,
            )
        elif method == "hardEM":
            self.train_EM_hard_log(
                inputs,
                num_iter=epochs,
                continue_training=continue_training,
                batch_size=batch_size,
            )
        else:
            raise ValueError("Invalid training method name")

    def inference(self, input_ids) -> list:
        return self.viterbi_log(input_ids)

    # ------------------------------------------------------------------
    # Supervised MLE (unchanged except vectorised normalisation)
    # ------------------------------------------------------------------
    def train_mle(self, inputs: Dataset):
        logger.info("Running MLE")
        assert not self.log_scale
        for sentence in tqdm(inputs, "MLE training", len(inputs)):
            input_ids = sentence["input_ids"]
            tags = sentence["tags"]
            self.transition_prob[0, tags[0] + 1] += 1

            for i in range(len(input_ids)):
                if i < len(input_ids) - 1:
                    self.transition_prob[tags[i] + 1, tags[i + 1] + 1] += 1
                self.emission_prob[tags[i], input_ids[i]] += 1

        self._normalize(self.transition_prob, self.epsilon)
        self._normalize(self.emission_prob, self.epsilon)

    def train_logmle(self, inputs: Dataset):
        """
        Supervised MLE, final parameters in log-space.
        """
        logger.info("Running log-scale MLE")
        assert not self.log_scale
        for sentence in tqdm(inputs, "Log-MLE training", len(inputs)):
            input_ids = sentence["input_ids"]
            tags = sentence["tags"]

            self.transition_prob[0, tags[0] + 1] += 1
            for i in range(len(input_ids)):
                if i < len(input_ids) - 1:
                    self.transition_prob[tags[i] + 1, tags[i + 1] + 1] += 1
                self.emission_prob[tags[i], input_ids[i]] += 1

        self._normalize_log_counts(self.transition_prob)
        self._normalize_log_counts(self.emission_prob)
        self.log_scale = True

    # ------------------------------------------------------------------
    # Batched Forward–Backward (used by EM and sEM)
    # ------------------------------------------------------------------
    def _forward_backward_batch(
        self,
        obs: torch.Tensor,      # (B, T_max)
        lengths: torch.Tensor,  # (B,)
    ):
        """
        Run forward–backward on a batch of sequences in log-space.

        Returns:
            gamma: (B, T_max, S) expected state occupancies (zero-padded)
            xi:    (B, T_max-1, S, S) expected transitions (zero-padded)
        """
        assert self.log_scale

        B, T_max = obs.shape
        S = self.num_states

        log_A = self.transition_prob            # (S+1, S+1)
        log_B = self.emission_prob             # (S, V)
        log_A_real = log_A[1:, 1:]             # (S, S)

        # mask for valid time steps
        time_ids = torch.arange(T_max, device=self.device).unsqueeze(0)  # (1, T_max)
        mask_time = time_ids < lengths.unsqueeze(1)                      # (B, T_max)

        # ----- Forward pass -----
        log_alpha = torch.full((B, T_max, S + 1), float("-inf"), device=self.device)

        # t = 0 for sequences with length > 0
        valid0 = lengths > 0
        if valid0.any():
            idx0 = valid0.nonzero(as_tuple=False).squeeze(1)
            o0 = obs[idx0, 0]  # (B0,)
            log_alpha[idx0, 0, 1:] = log_A[0, 1:] + log_B[:, o0].T

        for t in range(1, T_max):
            valid_t = lengths > t
            if not valid_t.any():
                continue
            idx = valid_t.nonzero(as_tuple=False).squeeze(1)
            prev_alpha = log_alpha[idx, t - 1, :]  # (Bv, S+1)
            # (Bv, S+1, S+1)
            log_scores = prev_alpha.unsqueeze(2) + log_A
            # transitions to real states 1..S, then emission:
            cur = torch.logsumexp(log_scores[:, :, 1:], dim=1) \
                  + log_B[:, obs[idx, t]].T  # (Bv, S)
            log_alpha[idx, t, 1:] = cur

        # ----- Backward pass -----
        log_beta = torch.full((B, T_max, S + 1), float("-inf"), device=self.device)

        # base cases: log_beta[b, length[b]-1, 1:] = 0
        last_idx = lengths - 1  # (B,)
        valid_last = lengths > 0
        if valid_last.any():
            idx_last = valid_last.nonzero(as_tuple=False).squeeze(1)
            t_last = last_idx[idx_last]
            log_beta[idx_last, t_last, 1:] = 0.0

        for t in range(T_max - 2, -1, -1):
            # we need log_beta[t] for sequences with length > t+1
            valid_t = lengths > (t + 1)
            if not valid_t.any():
                continue
            idx = valid_t.nonzero(as_tuple=False).squeeze(1)

            emit_next = log_B[:, obs[idx, t + 1]].T           # (Bv, S)
            beta_t1 = log_beta[idx, t + 1, 1:]                # (Bv, S)
            log_future = emit_next + beta_t1                  # (Bv, S)

            # (Bv, S, S): A_real[i,j] + future_j
            log_scores = log_A_real.unsqueeze(0) + log_future.unsqueeze(1)
            new_beta = torch.logsumexp(log_scores, dim=2)     # (Bv, S)

            log_beta[idx, t, 1:] = new_beta

        # ----- Gamma and Xi -----
        # gamma[b,t,s] ∝ alpha[b,t,s] * beta[b,t,s]
        log_unnorm_gamma = log_alpha[:, :, 1:] + log_beta[:, :, 1:]  # (B, T_max, S)

        # mask invalid time steps with a large negative constant
        log_unnorm_gamma = torch.where(
            mask_time.unsqueeze(-1),
            log_unnorm_gamma,
            torch.tensor(self._FNEG, device=self.device),
        )

        log_Z = torch.logsumexp(log_unnorm_gamma, dim=-1, keepdim=True)  # (B, T, 1)
        log_gamma = log_unnorm_gamma - log_Z
        gamma = torch.exp(log_gamma) * mask_time.unsqueeze(-1).float()  # (B, T, S)

        # xi[b,t,i,j] ∝ alpha[b,t,i] * A[i,j] * B[j,o_{t+1}] * beta[b,t+1,j]
        if T_max > 1:
            log_alpha_t = log_alpha[:, :-1, 1:]  # (B, T-1, S)
            log_beta_t1 = log_beta[:, 1:, 1:]    # (B, T-1, S)
            emit_next = log_B[:, obs[:, 1:]].permute(1, 2, 0)  # (B, T-1, S)

            term_j = emit_next + log_beta_t1  # (B, T-1, S)

            # (B, T-1, S, S)
            log_unnorm_xi = (
                log_alpha_t.unsqueeze(3)
                + log_A_real.view(1, 1, S, S)
                + term_j.unsqueeze(2)
            )

            # mask for valid transitions: t < lengths-1
            mask_pair = time_ids[:, :-1] < (lengths.unsqueeze(1) - 1)
            log_unnorm_xi = torch.where(
                mask_pair.unsqueeze(-1).unsqueeze(-1),
                log_unnorm_xi,
                torch.tensor(self._FNEG, device=self.device),
            )

            log_Z_xi = torch.logsumexp(
                log_unnorm_xi, dim=(2, 3), keepdim=True
            )  # (B, T-1, 1, 1)
            log_xi = log_unnorm_xi - log_Z_xi
            xi = (
                torch.exp(log_xi)
                * mask_pair.unsqueeze(-1).unsqueeze(-1).float()
            )  # (B, T-1, S, S)
        else:
            xi = torch.zeros(B, 0, S, S, device=self.device)

        return gamma, xi

    # ------------------------------------------------------------------
    # EM (batched)
    # ------------------------------------------------------------------
    def train_EM_log(
        self,
        inputs: Dataset,
        num_iter: int = 5,
        initial_guesses=None,
        continue_training=False,
        batch_size: int = 32,
    ):
        """
        Standard EM for HMM in log-space, with batched forward–backward.
        """
        if not self.log_scale:
            self.logify()

        if not continue_training:
            if initial_guesses is None:
                self.reset()
            else:
                A, B = initial_guesses
                self.transition_prob = A.to(self.device)
                self.emission_prob = B.to(self.device)
                self.log_scale = True

        S = self.num_states
        V = self.num_obs

        for _ in range(num_iter):
            expected_emissions = torch.zeros(S, V, device=self.device)
            expected_transitions = torch.zeros(S, S, device=self.device)
            expected_state_counts = torch.zeros(S, device=self.device)
            expected_initial = torch.zeros(S, device=self.device)

            for obs, lengths in self._batchify_sentences(inputs, batch_size):
                gamma, xi = self._forward_backward_batch(obs, lengths)

                # sum over batch and time
                expected_state_counts += gamma.sum(dim=(0, 1))
                expected_transitions += xi.sum(dim=(0, 1))
                expected_initial += gamma[:, 0, :].sum(dim=0)

                # emissions: flatten valid positions
                B_, T_max = obs.shape
                time_ids = torch.arange(T_max, device=self.device).unsqueeze(0)
                mask_time = time_ids < lengths.unsqueeze(1)  # (B, T)

                obs_flat = obs[mask_time]                  # (N_valid,)
                gamma_valid = gamma[mask_time]            # (N_valid, S)

                expected_emissions.scatter_add_(
                    1,
                    obs_flat.unsqueeze(0).expand(S, -1),
                    gamma_valid.T,
                )

            # M-step: build counts and convert to log-probs
            trans_counts = torch.full_like(self.transition_prob, self.epsilon)
            trans_counts[:, 0] = 0.0
            trans_counts[0, 1:] += expected_initial
            trans_counts[1:, 1:] += expected_transitions

            emis_counts = torch.full_like(self.emission_prob, self.epsilon)
            emis_counts += expected_emissions

            self.transition_prob.copy_(trans_counts)
            self.emission_prob.copy_(emis_counts)
            self.logify()

    # ------------------------------------------------------------------
    # Hard EM (Viterbi EM) with batched DP
    # ------------------------------------------------------------------
    def viterbi_log(self, input_ids):
        """
        Single-sequence Viterbi in log-space (original API).
        """
        assert self.log_scale
        T = len(input_ids)
        S = self.num_states

        log_pi = self.transition_prob[0, 1:]      # (S,)
        log_A = self.transition_prob[1:, 1:]      # (S, S)
        log_B = self.emission_prob               # (S, V)

        obs = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        V = torch.full((T, S), float("-inf"), device=self.device)
        ptr = torch.zeros((T, S), dtype=torch.long, device=self.device)

        V[0] = log_pi + log_B[:, obs[0]]
        for t in range(1, T):
            scores = V[t - 1].unsqueeze(1) + log_A  # (S, S)
            best_prev_score, best_prev_state = scores.max(dim=0)
            V[t] = best_prev_score + log_B[:, obs[t]]
            ptr[t] = best_prev_state

        last_state = torch.argmax(V[T - 1]).item()
        path = [last_state]
        for t in range(T - 1, 0, -1):
            last_state = ptr[t, last_state].item()
            path.append(last_state)
        path.reverse()
        return path

    def _viterbi_log_batch(
        self,
        obs: torch.Tensor,      # (B, T_max)
        lengths: torch.Tensor,  # (B,)
    ) -> List[List[int]]:
        """
        Batched Viterbi for use in hard EM.

        Returns:
            list of length B, each element is a list[int] of length lengths[b]
        """
        assert self.log_scale

        B, T_max = obs.shape
        S = self.num_states

        log_pi = self.transition_prob[0, 1:]      # (S,)
        log_A = self.transition_prob[1:, 1:]      # (S, S)
        log_B = self.emission_prob               # (S, V)

        V = torch.full((B, T_max, S), float("-inf"), device=self.device)
        ptr = torch.zeros((B, T_max, S), dtype=torch.long, device=self.device)

        # t = 0
        valid0 = lengths > 0
        if valid0.any():
            idx0 = valid0.nonzero(as_tuple=False).squeeze(1)
            o0 = obs[idx0, 0]  # (B0,)
            V[idx0, 0, :] = log_pi + log_B[:, o0].T  # (B0, S)

        # t >= 1
        for t in range(1, T_max):
            valid_t = lengths > t
            if not valid_t.any():
                continue
            idx = valid_t.nonzero(as_tuple=False).squeeze(1)
            prev_V = V[idx, t - 1, :]             # (Bv, S)
            scores = prev_V.unsqueeze(2) + log_A  # (Bv, S, S)
            best_prev_score, best_prev_state = scores.max(dim=1)  # (Bv, S)
            V[idx, t, :] = best_prev_score + log_B[:, obs[idx, t]].T
            ptr[idx, t, :] = best_prev_state

        # backtrace
        paths: List[List[int]] = []
        batch_indices = torch.arange(B, device=self.device)
        last_t = lengths - 1
        last_scores = V[batch_indices, last_t, :]         # (B, S)
        last_states = torch.argmax(last_scores, dim=1)    # (B,)

        for b in range(B):
            L = lengths[b].item()
            if L <= 0:
                paths.append([])
                continue
            s = last_states[b].item()
            path = [s]
            for t in range(L - 1, 0, -1):
                s = ptr[b, t, s].item()
                path.append(s)
            path.reverse()
            paths.append(path)

        return paths

    def train_EM_hard_log(
        self,
        inputs: Dataset,
        num_iter: int = 10,
        initial_guesses=None,
        continue_training=False,
        batch_size: int = 32,
    ):
        """
        Viterbi EM (hard EM), using batched Viterbi DP.
        """
        if not continue_training:
            if initial_guesses:
                self.transition_prob, self.emission_prob = initial_guesses
            else:
                self.reset()
        if not self.log_scale:
            self.logify()

        S = self.num_states
        V = self.num_obs

        for _ in range(num_iter):
            emis_counts = torch.zeros(S, V, device=self.device)
            trans_counts = torch.zeros(S + 1, S + 1, device=self.device)

            for obs, lengths in self._batchify_sentences(inputs, batch_size):
                paths = self._viterbi_log_batch(obs, lengths)  # list of length B

                B, T_max = obs.shape
                for b in range(B):
                    L = lengths[b].item()
                    if L == 0:
                        continue
                    seq = obs[b, :L]
                    path = paths[b]

                    # emissions
                    for t in range(L):
                        emis_counts[path[t], seq[t]] += 1

                    # transitions: start -> first state
                    trans_counts[0, path[0] + 1] += 1
                    # state -> state
                    for t in range(L - 1):
                        trans_counts[path[t] + 1, path[t + 1] + 1] += 1

            self.emission_prob = emis_counts
            self.transition_prob = trans_counts
            self.logify()

    # ------------------------------------------------------------------
    # Stepwise (online) EM (batched)
    # ------------------------------------------------------------------
    def train_sEM(
        self,
        inputs: Dataset,
        num_iter: int = 30,
        eta_fn: Callable[[int], float] = lambda k: 0.8,
        initial_guesses=None,
        continue_training=False,
        batch_size: int = 32,
    ):
        """
        Stepwise/online EM in log-space, using batched forward–backward.

        We maintain running expectations and update them per batch with step size
        eta_fn(k), where k is the number of sentences processed so far.
        We rebuild the full parameters and logify() once per epoch.
        """
        S = self.num_states
        V = self.num_obs

        if not continue_training:
            if initial_guesses is None:
                self.reset()  # creates log-scale params
            else:
                A, B = initial_guesses
                self.transition_prob = A.to(self.device)
                self.emission_prob = B.to(self.device)
                self.log_scale = True

        if not self.log_scale:
            self.logify()

        k = self.cnt  # global counter over sentences

        # running expectations
        expected_transitions = torch.zeros(S, S, device=self.device)
        expected_state_counts = torch.zeros(S, device=self.device)
        expected_emissions = torch.zeros(S, V, device=self.device)
        expected_initial = torch.zeros(S, device=self.device)

        for _ in range(num_iter):
            # per-epoch loop over batches
            for obs, lengths in self._batchify_sentences(inputs, batch_size):
                gamma, xi = self._forward_backward_batch(obs, lengths)

                # aggregate per batch
                ex_state_counts = gamma.sum(dim=(0, 1))        # (S,)
                ex_transitions = xi.sum(dim=(0, 1))            # (S, S)
                ex_initial = gamma[:, 0, :].sum(dim=0)         # (S,)

                B_, T_max = obs.shape
                time_ids = torch.arange(T_max, device=self.device).unsqueeze(0)
                mask_time = time_ids < lengths.unsqueeze(1)    # (B, T)

                obs_flat = obs[mask_time]                     # (N_valid,)
                gamma_valid = gamma[mask_time]                # (N_valid, S)

                ex_emissions = torch.zeros(S, V, device=self.device)
                ex_emissions.scatter_add_(
                    1,
                    obs_flat.unsqueeze(0).expand(S, -1),
                    gamma_valid.T,
                )

                # step-size based on number of sentences seen so far
                k += B_
                rate = eta_fn(k)

                expected_emissions = (1.0 - rate) * expected_emissions + rate * ex_emissions
                expected_state_counts = (1.0 - rate) * expected_state_counts + rate * ex_state_counts
                expected_initial = (1.0 - rate) * expected_initial + rate * ex_initial
                expected_transitions = (1.0 - rate) * expected_transitions + rate * ex_transitions

            # after each epoch, rebuild parameters once
            trans_counts = torch.full_like(self.transition_prob, self.epsilon)
            trans_counts[:, 0] = 0.0
            trans_counts[0, 1:] += expected_initial
            trans_counts[1:, 1:] += expected_transitions

            emis_counts = torch.full_like(self.emission_prob, self.epsilon)
            emis_counts += expected_emissions

            self.transition_prob.copy_(trans_counts)
            self.emission_prob.copy_(emis_counts)
            self.logify()

        self.cnt = k  # update global counter

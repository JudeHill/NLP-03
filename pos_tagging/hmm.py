import logging
import math
from typing import Callable

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

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
        self.num_states = num_states
        self.num_obs = num_obs
        # Initialized to epsilon, so allowing unseen transition/emission to have p>0
        A = torch.rand(self.num_states + 1, self.num_states + 1, device=self.device)
        A[:, 0] = 0.0
        B = torch.rand(self.num_states, self.num_obs, device=self.device)
        self.transition_prob = A
        self.emission_prob = B
        self.logify()
 
        # TODO: optimize training by using UNK token

    def reset(self):
        self.transition_prob = torch.full(
            [self.num_states + 1, self.num_states + 1], self.epsilon, device=self.device
        )
        self.transition_prob[:, 0] = 0.0
        self.emission_prob = torch.full([self.num_states, self.num_obs], self.epsilon,device=self.device)
        self.log_scale = False
        self.cnt = 0
        self.emission_lookup = {}
        self.emission_index = 0
        self.logify()

    def train(
        self,
        inputs: Dataset,
        epochs: int = 5,
        method: str = "mle",
        continue_training=False,
    ) -> None:
        if method == "mle":
            self.train_logmle(inputs)
        elif method == "EM":
            self.train_EM_log(
                inputs, num_iter=epochs, continue_training=continue_training
            )
        elif method == "sEM":
            self.train_sEM(
                inputs,
                num_iter=epochs,
                eta_fn=lambda k: (k + 2) ** (-1.0),
                continue_training=continue_training,
            )
        elif method == "hardEM":
            self.train_EM_hard_log(
                inputs, num_iter=epochs, continue_training=continue_training
            )
        else:
            raise ValueError("Invalid training method name")

    def inference(self, input_ids) -> list:
        return self.viterbi_log(input_ids)
    
    def logify(self):
        self._normalize_log(self.transition_prob)
        self._normalize_log(self.emission_prob)
        self.log_scale = True

    @staticmethod
    def _normalize(mat):
        for i in range(mat.size(0)):
            if torch.sum(mat[i]) == 0:
                continue
            mat[i] = mat[i] / torch.sum(mat[i])
        return mat

    @staticmethod
    def _log_normalize(log_matrix):
        return log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)

    @staticmethod
    def _normalize_log(mat):
        for i in range(mat.size(0)):
            if torch.sum(mat[i]) == 0:
                continue
            mat[i] = torch.log(mat[i]) - torch.log(torch.sum(mat[i]))
        return mat

    def train_mle(self, inputs: Dataset):
        """
        Supervised training by MLE
        """
        logger.info("Running MLE")
        assert not self.log_scale
        for sentence in tqdm(inputs, "MLE training", len(inputs)):
            # Tokens should have been tokenized
            input_ids = sentence["input_ids"]
            # UPoS or XPoS should have been mapped to integers
            tags = sentence["tags"]
            self.transition_prob[0, tags[0] + 1] += 1

            for i in range(len(input_ids)):
                if i < len(input_ids) - 1:
                    self.transition_prob[tags[i] + 1, tags[i + 1] + 1] += 1
                self.emission_prob[tags[i], input_ids[i]] += 1

        self.transition_prob = self._normalize(self.transition_prob)
        self.emission_prob = self._normalize(self.emission_prob)

    def train_logmle(self, inputs: Dataset):
        """Train with MLE algorithm using log likelihood to avoid underflow"""
        logger.info("Running log-scale MLE")
        assert not self.log_scale
        for sentence in tqdm(inputs, "Log-MLE training", len(inputs)):
            # Tokens should have been tokenized
            input_ids = sentence["input_ids"]
            # UPoS or XPoS should have been mapped to integers
            tags = sentence["tags"]

            # Update initial probabilities
            self.transition_prob[0, tags[0] + 1] += 1

            for i in range(len(input_ids)):
                # Update transition probabilities
                if i < len(input_ids) - 1:
                    self.transition_prob[tags[i] + 1, tags[i + 1] + 1] += 1

                # Update emission probabilities
                self.emission_prob[tags[i], input_ids[i]] += 1

        self.transition_prob = self._normalize_log(self.transition_prob)
        self.emission_prob = self._normalize_log(self.emission_prob)
        self.log_scale = True

    def train_EM_log(
        self,
        inputs: Dataset,
        num_iter: int = 5,
        initial_guesses=None,
        continue_training=False,
    ):  
        """
        Train an HMM with the standard EM algorithm
        """
        if not self.log_scale:
            self.logify()
            self.log_scale = True
        if not continue_training:
            if initial_guesses is None:
                self.reset()
            else:
                A, B = initial_guesses
                self.transition_prob = A
                self.emission_prob = B

        log_A, log_B = self.transition_prob, self.emission_prob
        for _ in range(num_iter):
            expected_emissions = torch.zeros(self.num_states, self.num_obs, device=self.device)
            expected_transitions = torch.zeros(self.num_states, self.num_states, device=self.device)
            expected_state_counts = torch.zeros(self.num_states, device=self.device)
            expected_initial = torch.zeros(self.num_states, device=self.device)
            for eg in inputs:
                
                obs = eg["input_ids"]
                n = len(obs)
                
                obs = torch.tensor(obs, dtype=torch.long, device=self.device)

                log_alpha = torch.full((n, self.num_states+1), float('-inf'), device=self.device)
                log_alpha[0, 1:] = log_A[0, 1:] + log_B[:, obs[0]]
                for t in range(1, n):
                    log_scores = log_alpha[t-1].unsqueeze(1) + log_A
                    log_alpha[t, 1:] = torch.logsumexp(log_scores[:, 1:], dim=0) + log_B[:, obs[t]]
                log_beta = torch.full((n, self.num_states+1), float('-inf'), device=self.device)
                log_beta[n-1, 1:] = 0.0 
                log_A_real = log_A[1:, 1:] 
                for t in range(n-2, -1, -1):
                    log_emit_next = log_B[:, obs[t+1]]       
                    log_future = log_emit_next + log_beta[t+1, 1:]    
                    log_scores = log_A_real + log_future.unsqueeze(0) 
                    log_beta[t, 1:] = torch.logsumexp(log_scores, dim=1)

                log_unnorm_gamma = log_alpha[:, 1:] + log_beta[:, 1:]
                log_Z = torch.logsumexp(log_unnorm_gamma, dim=1, keepdim=True)
                log_gamma = log_unnorm_gamma - log_Z
            
                log_alpha_real = log_alpha[:, 1:]       
                log_beta_real  = log_beta[:, 1:]       

                log_alpha_t = log_alpha_real[:-1]     
                log_beta_t1 = log_beta_real[1:]       

        
                log_emit_next = log_B[:, obs[1:]].T   
                log_A_real = log_A[1:, 1:] 
                log_unnorm_xi = (
                    log_alpha_t.unsqueeze(2)         
                    + log_A_real.unsqueeze(0)              
                    + log_emit_next.unsqueeze(1)       
                    + log_beta_t1.unsqueeze(1)         
                )                                     

                log_Z = torch.logsumexp(log_unnorm_xi, dim=(1, 2), keepdim=True)
                log_xi = log_unnorm_xi - log_Z      
                gamma, xi = torch.exp(log_gamma), torch.exp(log_xi)
                expected_state_counts += gamma.sum(dim=0)
                expected_transitions += xi.sum(dim=0)

                expected_initial += gamma[0]
                
                expected_emissions.scatter_add_(
                    1, # dimension
                    obs.unsqueeze(0).expand(self.num_states, -1),
                    gamma.T
                )

            # M step (done after all inputs processed)
            # After computing expected_initial (counts) in linear space:
            trans_counts = torch.full_like(self.transition_prob, self.epsilon)
            trans_counts[:, 0] = 0.0
            trans_counts[0, 1:] += expected_initial  # shape (S,)
            trans_counts[1:, 1:] += expected_transitions  # shape (S, S)
            emis_counts = torch.full_like(self.emission_prob, self.epsilon)
            emis_counts += expected_emissions  # add expected emission counts
            self.transition_prob.copy_(trans_counts)
            self.emission_prob.copy_(emis_counts)
            self.logify()


   
    def lookup_emission(self, emission: str) -> int:
        if emission in self.emission_lookup:
            return self.emission_lookup[emission]
        else:
            self.emission_lookup[emission] = self.emission_index
            self.emission_index += 1
            return self.emission_index

    def train_EM_hard_log(
        self,
        inputs: Dataset,
        num_iter: int = 10,
        initial_guesses=None,
        continue_training=False,
    ):
        """
        Train an HMM with the hard EM algorithm (also called Viterbi EM)
        """
        
        
        if not continue_training:
            if initial_guesses:
                self.transition_prob, self.emission_prob = initial_guesses
            else:
                self.reset()
        if not self.log_scale:
            self.logify()
            self.log_scale = True
        for _ in range(num_iter):
            emis_counts = torch.zeros(self.num_states, self.num_obs, device=self.device)
            trans_counts = torch.zeros(self.num_states+1, self.num_states+1, device=self.device)
            for eg in inputs:
                obs = eg["input_ids"]
                obs = torch.tensor(obs, dtype=torch.long, device=self.device)
                path = self.viterbi_log(obs)
                T = obs.shape[0]
                for t in range(T):
                    emis_counts[path[t], obs[t]] += 1
                # start → first state
                trans_counts[0, path[0] + 1] += 1
                # state → state
                for t in range(T - 1):
                    trans_counts[path[t] + 1, path[t+1] + 1] += 1

            self.emission_prob, self.transition_prob = emis_counts, trans_counts
            self.logify()

    def train_sEM(
        self,
        inputs: Dataset,
        num_iter: int = 30,
        eta_fn: Callable[[int], float] = lambda k: 0.8,
        initial_guesses=None,
        continue_training=False,
    ):
        """
        Train an HMM with a stepwise online EM algorithm (in log-space).

        Parameters are stored in log-space in self.transition_prob (shape (S+1,S+1))
        and self.emission_prob (shape (S,V)).

        For each sentence, we:
        1) run forward–backward in log-space using current parameters
        2) compute expected sufficient statistics (gamma, xi)
        3) update running expectations with a step size eta_fn(k)
        4) rebuild full count matrices from expectations and call `logify()`
        """

        S = self.num_states
        V = self.num_obs

        # ----- Initialization of parameters -----
        if not continue_training:
            if initial_guesses is None:
                self.reset()  # also creates log-scale params via logify()
            else:
                A, B = initial_guesses
                self.transition_prob = A.to(self.device)
                self.emission_prob = B.to(self.device)
                self.log_scale = True

        if not self.log_scale:
            self.logify()
            self.log_scale = True

        # Global update counter for step-size schedule
        # (you already have self.cnt; we’ll use that)
        k = self.cnt

        for _ in range(num_iter):
            # Running expectations for this epoch (in linear space)
            expected_transitions = torch.zeros(S, S, device=self.device)
            expected_state_counts = torch.zeros(S, device=self.device)
            expected_emissions = torch.zeros(S, V, device=self.device)
            expected_initial = torch.zeros(S, device=self.device)

            for eg in inputs:
                k += 1
                rate = eta_fn(k)

                log_A = self.transition_prob        # (S+1, S+1), log-space
                log_B = self.emission_prob          # (S, V),     log-space

                obs_list = eg["input_ids"]
                n = len(obs_list)
                obs = torch.tensor(obs_list, dtype=torch.long, device=self.device)
                log_alpha = torch.full((n, S + 1), float('-inf'), device=self.device)
                log_alpha[0, 1:] = log_A[0, 1:] + log_B[:, obs[0]]

                for t in range(1, n):
                    log_scores = log_alpha[t - 1].unsqueeze(1) + log_A  # (S+1, S+1)
                    log_alpha[t, 1:] = torch.logsumexp(log_scores[:, 1:], dim=0) \
                                    + log_B[:, obs[t]]
                log_beta = torch.full((n, S + 1), float('-inf'), device=self.device)
                log_beta[n - 1, 1:] = 0.0  
                log_A_real = log_A[1:, 1:]  # (S, S)

                for t in range(n - 2, -1, -1):
                    log_emit_next = log_B[:, obs[t + 1]]           # (S,)
                    log_future = log_emit_next + log_beta[t + 1, 1:]  # (S,)
                    log_scores = log_A_real + log_future.unsqueeze(0)  # (S, S)
                    log_beta[t, 1:] = torch.logsumexp(log_scores, dim=1)

                log_unnorm_gamma = log_alpha[:, 1:] + log_beta[:, 1:]  # (n, S)
                log_Z = torch.logsumexp(log_unnorm_gamma, dim=1, keepdim=True)
                log_gamma = log_unnorm_gamma - log_Z
                gamma = torch.exp(log_gamma)  # (n, S) in linear space

                log_alpha_real = log_alpha[:, 1:]  # (n, S)
                log_beta_real = log_beta[:, 1:]    # (n, S)

                log_alpha_t = log_alpha_real[:-1]  # (n-1, S)
                log_beta_t1 = log_beta_real[1:]    # (n-1, S)
                log_emit_next = log_B[:, obs[1:]].T  # (n-1, S)

                log_unnorm_xi = (
                    log_alpha_t.unsqueeze(2)       # (n-1, S, 1)
                    + log_A_real.unsqueeze(0)      # (1, S, S)
                    + log_emit_next.unsqueeze(1)   # (n-1, 1, S)
                    + log_beta_t1.unsqueeze(1)     # (n-1, 1, S)
                )  # -> (n-1, S, S)

                log_Z_xi = torch.logsumexp(log_unnorm_xi, dim=(1, 2), keepdim=True)
                log_xi = log_unnorm_xi - log_Z_xi
                xi = torch.exp(log_xi)  # (n-1, S, S) in linear space

                ex_state_counts = gamma.sum(dim=0)      # (S,)
                ex_transitions = xi.sum(dim=0)          # (S, S)
                ex_initial = gamma[0]                   # (S,)

                ex_emissions = torch.zeros(S, V, device=self.device)
                ex_emissions.scatter_add_(
                    1,  # dim
                    obs.unsqueeze(0).expand(S, -1),
                    gamma.T
                )

                expected_emissions = (1.0 - rate) * expected_emissions + rate * ex_emissions
                expected_state_counts = (1.0 - rate) * expected_state_counts + rate * ex_state_counts
                expected_initial = (1.0 - rate) * expected_initial + rate * ex_initial
                expected_transitions = (1.0 - rate) * expected_transitions + rate * ex_transitions
                
                trans_counts = torch.full_like(self.transition_prob, self.epsilon)
                trans_counts[:, 0] = 0.0  
                trans_counts[0, 1:] += expected_initial
                trans_counts[1:, 1:] += expected_transitions
                emis_counts = torch.full_like(self.emission_prob, self.epsilon)
                emis_counts += expected_emissions

                self.transition_prob.copy_(trans_counts)
                self.emission_prob.copy_(emis_counts)
                self.logify()  



    def viterbi(self, input_ids):
        """Run Viterbi algorithm in probability space (vectorized over states)."""
        assert not self.log_scale

        T = len(input_ids)
        S = self.num_states

        pi = self.transition_prob[0, 1:]        # (S,)
        A  = self.transition_prob[1:, 1:]       # (S, S)
        B  = self.emission_prob                # (S, V)

        obs = torch.tensor(input_ids, dtype=torch.long, device=self.device)

        V = torch.zeros(T, S, device=self.device)
        ptr = torch.zeros(T, S, dtype=torch.long, device=self.device)

        V[0] = pi * B[:, obs[0]]   # (S,)

        for t in range(1, T):
            scores = V[t-1].unsqueeze(1) * A    # (S, S)
            best_prev_prob, best_prev_state = scores.max(dim=0)  # both (S,)
            V[t] = best_prev_prob * B[:, obs[t]]  # (S,)
            ptr[t] = best_prev_state              # (S,)

        last_state = torch.argmax(V[T-1]).item()

        path = [last_state]
        for t in range(T-1, 0, -1):
            last_state = ptr[t, last_state].item()
            path.append(last_state)
        path.reverse()
        return path


    def viterbi_log(self, input_ids):
        """Run Viterbi algorithm with log-scale probabilities (vectorized over states)."""
        assert self.log_scale

        T = len(input_ids)
        S = self.num_states

        log_pi = self.transition_prob[0, 1:]      # (S,)
        log_A  = self.transition_prob[1:, 1:]     # (S, S)
        log_B  = self.emission_prob              # (S, V)

        obs = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        V = torch.full((T, S), float("-inf"), device=self.device)
        ptr = torch.zeros((T, S), dtype=torch.long, device=self.device)
        V[0] = log_pi + log_B[:, obs[0]]   # (S,)
        for t in range(1, T):
            scores = V[t-1].unsqueeze(1) + log_A   # (S, S)
            best_prev_score, best_prev_state = scores.max(dim=0)  # both (S,)
            V[t] = best_prev_score + log_B[:, obs[t]]  # (S,)
            ptr[t] = best_prev_state                   # (S,)

        last_state = torch.argmax(V[T-1]).item()
        path = [last_state]
        for t in range(T-1, 0, -1):
            last_state = ptr[t, last_state].item()
            path.append(last_state)

        path.reverse()

        return path



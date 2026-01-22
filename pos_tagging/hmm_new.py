import logging
import math
from typing import Callable

import numpy as np
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
        # gpu_check.check()
        self.cnt = 0
        self.num_states = num_states
        self.num_obs = num_obs
        self.epsilon = 1e-6
        # Initialized to epsilon, so allowing unseen transition/emission to have p>0
        A = torch.rand(self.num_states + 1, self.num_states + 1, device=self.device)
        A[:, 0] = 0.0
        B = torch.rand(self.num_states, self.num_obs, device=self.device)
        self.transition_prob = A
        self.emission_prob = B
        self.log_scale = False
        self.convert_log_space()
 
        # TODO: optimize training by using UNK token

    def reset(self):
        A = torch.rand(self.num_states + 1, self.num_states + 1, device=self.device)
        A[:, 0] = 0.0
        B = torch.rand(self.num_states, self.num_obs, device=self.device)
        self.transition_prob = A
        self.emission_prob = B
        self.log_scale = False
        self.convert_log_space()

    def train(
        self,
        inputs: Dataset,
        epochs: int = 5,
        method: str = "mle",
        continue_training=False,
        alpha=0.8
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
                eta_fn=lambda k: (k + 2) ** (-alpha),
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
    
    def convert_log_space(self):
        if self.log_scale:
            return
        # if they’re counts, convert to prob then log-normalize:
        self.transition_prob = self._normalize_log(self.transition_prob)
        self.emission_prob   = self._normalize_log(self.emission_prob)
        self.log_scale = True

    def convert_linear_space(self):
        """
        Converts transition and emission matrices from log-probabilities 
        back to standard probability space [0, 1].
        """
        if not self.log_scale:
            return  # Already in linear space
            
        # Use np.exp to reverse the log operation
        self.transition_prob = np.exp(self.transition_prob)
        self.emission_prob   = np.exp(self.emission_prob)
        self.emission_prob = self._normalize(self.emission_prob)
        self.transition_prob = self._normalize(self.transition_prob)
        self.log_scale = False




    def counts_to_log_probs(self, counts):
        """Convert count matrix to log probabilities"""
        return self._normalize_log(counts)


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
        if self.log_scale:
            self.convert_linear_space()
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
            self.convert_log_space()
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
            # M step
            trans_counts = torch.full_like(self.transition_prob, self.epsilon)
            trans_counts[:, 0] = 0.0
            trans_counts[0, 1:] += expected_initial
            trans_counts[1:, 1:] += expected_transitions

            emis_counts = torch.full_like(self.emission_prob, self.epsilon)
            emis_counts += expected_emissions

            self.transition_prob.copy_(trans_counts)
            self.emission_prob.copy_(emis_counts)  # ← FIX: use emis_counts, not expected_emissions
            self.convert_log_space()

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
            self.convert_log_space()
            self.log_scale = True
        for _ in range(num_iter):
            emis_counts = torch.full((self.num_states, self.num_obs), self.epsilon, device=self.device)
            trans_counts = torch.full((self.num_states+1, self.num_states+1), self.epsilon, device=self.device)
            trans_counts[:, 0] = 0.0
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

            self.transition_prob = trans_counts
            self.emission_prob = emis_counts
            self.convert_log_space()

    def train_sEM(
        self,
        inputs: Dataset,
        num_iter: int = 30,
        eta_fn: Callable[[int], float] = lambda k: 0.8,
        initial_guesses=None,
        continue_training=False,
    ):
        """
        Stepwise / online EM for HMM using:
        (1) log-space forward-backward (prevents underflow)
        (2) EMA over expected COUNT statistics (prevents collapse from mixing probabilities)

        Dummy/start state handling:
        - Column 0 is the dummy state.
        - Enforce P(any_state -> dummy) = 0 exactly.
            => trans_counts[:,0] = 0 in count-space
            => log_A[:,0] = -inf in log-space
        """

        S = self.num_states
        V = self.num_obs
        eps = self.epsilon
        neg_inf = float("-inf")

        # ----- Initialize parameters -----
        if not continue_training:
            if initial_guesses is None:
                self.reset()  # random + logify()
            else:
                A, B = initial_guesses
                self.transition_prob = A.to(self.device)
                self.emission_prob = B.to(self.device)
                self.log_scale = True

        if not self.log_scale:
            self.convert_log_space()

        # Hard-enforce dummy-column is impossible in log-space
        with torch.no_grad():
            self.transition_prob[:, 0] = neg_inf

        # ----- Initialize EMA "counts" from current params -----
        # We'll keep EMAs in linear space.
        with torch.no_grad():
            A_prob = torch.exp(self.transition_prob)  # (S+1, S+1), dummy col becomes 0
            B_prob = torch.exp(self.emission_prob)    # (S, V)

            ema_init = torch.clamp(A_prob[0, 1:].clone(), min=eps)     # (S,)
            ema_trans = torch.clamp(A_prob[1:, 1:].clone(), min=eps)   # (S,S)
            ema_emit = torch.clamp(B_prob.clone(), min=eps)            # (S,V)

        k = 0

        for _ in range(num_iter):
            for eg in inputs:
                k += 1
                rate = float(eta_fn(k))

                log_A = self.transition_prob  # (S+1, S+1)
                log_B = self.emission_prob    # (S, V)

                obs_list = eg["input_ids"]
                n = len(obs_list)
                obs = torch.tensor(obs_list, dtype=torch.long, device=self.device)

                # ===== Forward (log-alpha) =====
                log_alpha = torch.full((n, S + 1), neg_inf, device=self.device)
                log_alpha[0, 1:] = log_A[0, 1:] + log_B[:, obs[0]]

                for t in range(1, n):
                    scores = log_alpha[t - 1].unsqueeze(1) + log_A  # (S+1, S+1)
                    log_alpha[t, 1:] = torch.logsumexp(scores[:, 1:], dim=0) + log_B[:, obs[t]]

                # ===== Backward (log-beta) =====
                log_beta = torch.full((n, S + 1), neg_inf, device=self.device)
                log_beta[n - 1, 1:] = 0.0  # log(1)

                log_A_real = log_A[1:, 1:]  # (S,S)
                for t in range(n - 2, -1, -1):
                    log_emit_next = log_B[:, obs[t + 1]]                 # (S,)
                    log_future = log_emit_next + log_beta[t + 1, 1:]     # (S,)
                    scores = log_A_real + log_future.unsqueeze(0)        # (S,S)
                    log_beta[t, 1:] = torch.logsumexp(scores, dim=1)

                # ===== Gamma =====
                log_gamma_unnorm = log_alpha[:, 1:] + log_beta[:, 1:]    # (n,S)
                log_gamma = log_gamma_unnorm - torch.logsumexp(
                    log_gamma_unnorm, dim=1, keepdim=True
                )
                gamma = torch.exp(log_gamma)  # (n,S)

                # ===== Xi =====
                log_alpha_real = log_alpha[:, 1:]  # (n,S)
                log_beta_real = log_beta[:, 1:]    # (n,S)

                log_alpha_t = log_alpha_real[:-1]           # (n-1,S)
                log_beta_t1 = log_beta_real[1:]             # (n-1,S)
                log_emit_next = log_B[:, obs[1:]].T         # (n-1,S)

                log_xi_unnorm = (
                    log_alpha_t.unsqueeze(2)        # (n-1,S,1)
                    + log_A_real.unsqueeze(0)       # (1,S,S)
                    + log_emit_next.unsqueeze(1)    # (n-1,1,S)
                    + log_beta_t1.unsqueeze(1)      # (n-1,1,S)
                )  # (n-1,S,S)

                log_xi = log_xi_unnorm - torch.logsumexp(log_xi_unnorm, dim=(1, 2), keepdim=True)
                xi = torch.exp(log_xi)  # (n-1,S,S)

                # ===== Expected counts for this sentence =====
                ex_init = gamma[0]         # (S,)
                ex_trans = xi.sum(dim=0)   # (S,S)

                ex_emit = torch.zeros(S, V, device=self.device)
                ex_emit.scatter_add_(
                    1,
                    obs.unsqueeze(0).expand(S, -1),
                    gamma.T
                )
    

                # ===== EMA over counts =====
                with torch.no_grad():
                    ema_init = (1.0 - rate) * ema_init + rate * ex_init
                    ema_trans = (1.0 - rate) * ema_trans + rate * ex_trans
                    ema_emit = (1.0 - rate) * ema_emit + rate * ex_emit

                    # Build full transition "count" matrix with dummy column forced to 0
                    trans_counts = torch.full((S + 1, S + 1), self.epsilon, device=self.device)
                    trans_counts[:, 0] = 0.0               # <-- EXACTLY ZERO into dummy
                    trans_counts[0, 1:] += ema_init
                    trans_counts[1:, 1:] += ema_trans

                    emis_counts = torch.full_like(self.emission_prob, self.epsilon) 
                    emis_counts += ema_emit # already positive

                    # Convert counts -> log-probs
                    new_log_A = self._normalize_log(trans_counts)
                    new_log_B = self._normalize_log(emis_counts)

                    # Hard-enforce dummy column is impossible (numerical safety)
                    new_log_A[:, 0] = neg_inf

                    self.transition_prob = new_log_A
                    self.emission_prob = new_log_B
                    self.log_scale = True  


    def viterbi(self, input_ids):
        """Run Viterbi algorithm in probability space (vectorized over states)."""
        if self.log_scale:
            return self.viterbi_log(input_ids)

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
        if not self.log_scale:
            self.convert_log_space()

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



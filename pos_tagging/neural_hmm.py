from pos_tagging.base import BaseUnsupervisedClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import pos_tagging.gpu_check as gpu_check



class NeuralHMMClassifier(nn.Module):

    def __init__(
        self,
        num_states: int,
        vocab_size: int,
        tag_emb_dim: int = 128,
        trans_hidden_dim: int = 128,
        
    ):
        """
        Initialize a neural parameterization of a Hidden Markov Model (HMM).

        This module defines learnable parameters for initial state probabilities,
        emission distributions, and transition probabilities using neural
        embeddings and linear projections.

        Args:
            num_states (int): Number of hidden states.
            vocab_size (int): Size of the observation vocabulary.
            tag_emb_dim (int, optional): Dimensionality of tag and word embeddings.
                Defaults to 128.
            trans_hidden_dim (int, optional): Dimensionality of the hidden
                representation used to parameterize state transition probabilities.
                Defaults to 128.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")
        self.init_logits = nn.Parameter(torch.zeros(num_states))
        self.tag_embed = nn.Embedding(num_states, tag_emb_dim)
        self.word_embed = nn.Embedding(vocab_size, tag_emb_dim)
        self.word_bias = nn.Parameter(
            torch.zeros(vocab_size)
        )
        self.trans_linear = nn.Linear(
            trans_hidden_dim, num_states * num_states
        )

        self.trans_query = nn.Parameter(torch.zeros(trans_hidden_dim))

        self.num_states = num_states
        self.vocab_size = vocab_size
        self.tag_emb_dim = tag_emb_dim
        self.trans_hidden_dim = trans_hidden_dim
        

    def compute_initial_log_probs(self) -> torch.Tensor:
        """
        Returns:
            log_pi: (K,) log initial state probabilities.
        """
        return F.log_softmax(self.init_logits, dim=0)

    def compute_transition_log_probs(self) -> torch.Tensor:
        """
        Returns:
            log_A: (K, K) log transition probabilities,
                   log_A[i, j] = log p(z_t=j | z_{t-1}=i).
        """

        h = self.trans_query
        logits = self.trans_linear(h)
        logits = logits.view(self.num_states, self.num_states)
        return F.log_softmax(logits, dim=1)
        

    def compute_emission_log_probs(self) -> torch.Tensor:
        """
        Returns:
            log_B: (K, V) log emission probs,
                   log_B[k, w] = log p(x_t=w | z_t=k).
        """
        U = self.tag_embed.weight      # (K, D)
        V = self.word_embed.weight     # (V, D)
        b = self.word_bias             # (V,)

        scores = U @ V.T + b           # (K, V)
        return F.log_softmax(scores, dim=1)
        ...
    
    # ---- HMM inference helpers ----

    def forward_algorithm(
        self,
        emissions: torch.Tensor,
        log_pi: torch.Tensor,
        log_A: torch.Tensor,
        log_B: torch.Tensor,
    ) -> torch.Tensor:

        """
        Compute log forward probabilities for a single observation sequence.

        This function implements the forward pass of the forward-backward
        algorithm in log space. For each time step t and state k, it computes

            log α_t(k) = log p(o_1, …, o_t, z_t = k),

        using the recursive relation

            log α_t = logsumexp(log α_{t-1} + log A) + log B(o_t).

        Args:
            emissions : torch.Tensor, shape (T,)
                Sequence of observed symbol indices.
            log_pi : torch.Tensor, shape (K,)
                Log initial state probabilities.
            log_A : torch.Tensor, shape (K, K)
                Log transition probability matrix.
            log_B : torch.Tensor, shape (K, V)
                Log emission probability matrix.

        Returns:
            log_alpha : torch.Tensor, shape (T, K)
                Log forward probabilities for each time step and state.
        """
        # Standard log-space forward recursion
        T = emissions.shape[0]
        K = log_pi.shape[0]
        log_alpha = emissions.new_full((T, K), fill_value=float("-inf"), dtype=log_pi.dtype)
        e0 = emissions[0]
        log_alpha[0] = log_pi + log_B[:, e0]
        for t in range(1, T):
            et = emissions[t]
            scores = log_alpha[t - 1].unsqueeze(1) + log_A  # (K, 1) + (K, K) -> (K, K)
            log_alpha[t] = torch.logsumexp(scores, dim=0) + log_B[:, et]

        return log_alpha
    
    def forward_backward(
        self,
        emissions: torch.Tensor,
        log_pi: torch.Tensor,
        log_A: torch.Tensor,
        log_B: torch.Tensor,
    ):
        """
        Run the forward-backward algorithm on a single observation sequence.

        This function combines `forward_algorithm` and `backward_algorithm` to
        compute posterior state and transition probabilities, as well as the
        log-likelihood of the sequence under the model. See the docstrings of
        `forward_algorithm` and `backward_algorithm` for detailed descriptions
        of the underlying recursions.

        Args:
            emissions : torch.Tensor, shape (T,)
                Sequence of observed symbol indices.
            log_pi : torch.Tensor, shape (K,)
                Log initial state probabilities.
            log_A : torch.Tensor, shape (K, K)
                Log transition probability matrix.
            log_B : torch.Tensor, shape (K, V)
                Log emission probability matrix.

        Returns:
            log_alpha : torch.Tensor, shape (T, K)
                Log forward probabilities.
            log_beta : torch.Tensor, shape (T, K)
                Log backward probabilities.
            log_gamma : torch.Tensor, shape (T, K)
                Log posterior state probabilities.
            log_xi : torch.Tensor, shape (T-1, K, K)
                Log posterior transition probabilities.
            log_likelihood : torch.Tensor
                Log-likelihood of the observation sequence.

        """
        T = emissions.shape[0]
        K = log_pi.shape[0]

        log_alpha = self.forward_algorithm(emissions, log_pi, log_A, log_B)  # (T, K)
        log_beta  = self.backward_algorithm(emissions, log_pi, log_A, log_B)  # (T, K)
        log_likelihood = torch.logsumexp(log_alpha[-1], dim=-1)  # scalar

        log_gamma_unnorm = log_alpha + log_beta  # (T, K)
        log_gamma = log_gamma_unnorm - torch.logsumexp(log_gamma_unnorm, dim=1, keepdim=True)
        # xi_t(i,j) ∝ α_t(i) + log_A[i,j] + log_B[j, x_{t+1}] + β_{t+1}(j)
        log_xi = emissions.new_empty((T - 1, K, K), dtype=log_pi.dtype)

        for t in range(T - 1):
            et1 = emissions[t + 1]
            log_xi_t = (
                log_alpha[t].unsqueeze(1)              # (K, 1)
                + log_A                                # (K, K)
                + log_B[:, et1].unsqueeze(0)           # (1, K)
                + log_beta[t + 1].unsqueeze(0)         # (1, K)
            )
            log_xi[t] = log_xi_t - torch.logsumexp(log_xi_t.view(-1), dim=0)

        return log_alpha, log_beta, log_gamma, log_xi, log_likelihood


    def forward(self, batch_emissions: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative average log-likelihood for a batch of sequences.

        For each sequence in the batch, this method computes the log-likelihood
        using the forward algorithm. See `forward_algorithm` for a detailed
        description of the underlying recursion.

        Args:
            batch_emissions : torch.Tensor, shape (B, T)
                Batch of observation sequences. All sequences must have the same
                length and contain no padding tokens.

        Returns:
            loss : torch.Tensor
                Scalar tensor equal to the negative average log-likelihood over
                the batch.
        """
        self.to(self.device)
        batch_emissions = batch_emissions.to(self.device)

        B, T = batch_emissions.shape

        # 1. Compute log parameters once per batch
        log_pi = self.compute_initial_log_probs()     # (K,)
        log_A  = self.compute_transition_log_probs()  # (K, K)
        log_B  = self.compute_emission_log_probs()    # (K, V)

        total_log_likelihood = torch.tensor(0.0, device=self.device)

        for b in range(B):
            emissions = batch_emissions[b]            # (T,)
            log_alpha = self.forward_algorithm(emissions, log_pi, log_A, log_B)

            # log p(x_1:T) = logsumexp over final log_alpha
            seq_log_likelihood = torch.logsumexp(log_alpha[-1], dim=-1)
            total_log_likelihood += seq_log_likelihood

        # Negative average log-likelihood over the batch
        loss = - total_log_likelihood / B
        return loss

    def backward_algorithm(
        self,
        emissions: torch.Tensor,
        log_pi: torch.Tensor,
        log_A: torch.Tensor,
        log_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log backward probabilities for a single observation sequence.

        This function implements the backward pass of the forward-backward
        algorithm in log space. For each time step t and state k, it computes

            log β_t(k) = log p(o_{t+1}, …, o_T | z_t = k),

        using the recursive relation

            log β_t = logsumexp(log A + log B(o_{t+1}) + log β_{t+1}).

        Args:    
            emissions : torch.Tensor, shape (T,)
                Sequence of observed symbol indices.
            log_pi : torch.Tensor, shape (K,)
                Log initial state probabilities. 
            log_A : torch.Tensor, shape (K, K)
                Log transition probability matrix.
            log_B : torch.Tensor, shape (K, V)
                Log emission probability matrix.

        Returns:
            log_beta : torch.Tensor, shape (T, K)
                Log backward probabilities for each time step and state.
        """
        T = emissions.shape[0]
        K = log_pi.shape[0]

        log_beta = emissions.new_full((T, K), fill_value=float("-inf"), dtype=log_pi.dtype)

        # Initialization: β_T-1(k) = 0 in log space
        log_beta[-1] = 0.0

        for t in range(T - 2, -1, -1):
            et1 = emissions[t + 1]  # x_{t+1}

            # log_beta[t, i] = logsum_j A[i,j] + log_B[j, x_{t+1}] + log_beta[t+1, j]
            scores = log_A + log_B[:, et1] + log_beta[t + 1]  # (K, K)
            # scores[i, j] = log_A[i, j] + log_B[j, x_{t+1}] + log_beta[t+1, j]

            log_beta[t] = torch.logsumexp(scores, dim=1)  # sum over j

        return log_beta

    def infer_posteriors(self, emissions: torch.Tensor):
        """
        Compute posterior state and transition probabilities for a single sequence.

        This is a convenience wrapper around `forward_backward` that returns the
        posterior state probabilities (γ), posterior transition probabilities (ξ),
        and the log-likelihood of the sequence under the model. See
        `forward_backward` for details of the underlying computations.

        Args:
            emissions : torch.Tensor, shape (T,)
                Sequence of observed symbol indices.

        Returns:
            log_gamma : torch.Tensor, shape (T, K)
                Log posterior state probabilities.
            log_xi : torch.Tensor, shape (T-1, K, K)
                Log posterior transition probabilities.
            log_likelihood : torch.Tensor
                Log-likelihood of the observation sequence.

        """

        self.to(self.device)
        emissions = emissions.to(self.device)

        log_pi = self.compute_initial_log_probs()
        log_A  = self.compute_transition_log_probs()
        log_B  = self.compute_emission_log_probs()

        _, _, log_gamma, log_xi, log_likelihood = self.forward_backward(
            emissions, log_pi, log_A, log_B
        )
        return log_gamma, log_xi, log_likelihood
    
    def inference(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform inference (tag prediction) for one or many sequences.

        Args:
            input_ids: (T,) or (B, T) tensor of word indices.

        Returns:
            preds: (T,) if input was (T,)
                   (B, T) if input was (B, T)
        """
        self.to(self.device)
        self.eval()

        with torch.no_grad():
            single = False

            # If input is a single sequence (T,), turn into (1, T)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                single = True

            input_ids = input_ids.to(self.device)
            B, T = input_ids.shape

            # Compute HMM parameters once
            log_pi = self.compute_initial_log_probs()
            log_A  = self.compute_transition_log_probs()
            log_B  = self.compute_emission_log_probs()

            all_preds = []

            for b in range(B):
                emissions = input_ids[b]  # (T,)

                # get posteriors
                _, _, log_gamma, _, _ = self.forward_backward(
                    emissions, log_pi, log_A, log_B
                )

                # predicted state = most likely hidden state per position
                preds_b = torch.argmax(log_gamma, dim=1)  # (T,)
                all_preds.append(preds_b)

            preds = torch.stack(all_preds, dim=0)  # (B, T)

            # If single sequence was provided, return shape (T,)
            if single:
                return preds[0]

            return preds

    

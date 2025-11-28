from pos_tagging.base import BaseUnsupervisedClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralHMMClassifier(nn.Module):

    def __init__(
        self,
        num_states: int,
        vocab_size: int,
        tag_emb_dim: int = 128,
        trans_hidden_dim: int = 128,
    ):
        super().__init__()

        self.init_logits = nn.Parameter(torch.zeros(num_states))
        self.tag_embed = nn.Embedding(num_states, tag_emb_dim)
        self.word_embed = nn.Embedding(vocab_size, tag_emb_dim)
        self.word_bias = nn.Parameter(
            torch.zeros(vocab_size)
        )
        self.trans_linear = nn.Linear(
            trans_hidden_dim, num_states * num_states
        )

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
        # 1. Compute a hidden vector from query (maybe identity)
        # h = self.trans_query            # (D,)
        # 2. Pass through linear -> (K*K,)
        # logits = self.trans_linear(h)   # (K*K,)
        # 3. Reshape to (K, K)
        # logits = logits.view(self.num_states, self.num_states)
        # 4. Row-wise log-softmax for transitions
        # log_A = F.log_softmax(logits, dim=1)
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
    
    # ---- HMM inference helpers (no neural nets here) ----

    def forward_algorithm(
        self,
        emissions: torch.Tensor,
        log_pi: torch.Tensor,
        log_A: torch.Tensor,
        log_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        emissions: (T,) integer word indices for a single sequence,
        log_pi:    (K,)
        log_A:     (K, K)
        log_B:     (K, V)

        Returns:
            log_alpha: (T, K) log forward probabilities.
        """
        # Standard log-space forward recursion
        ...
    
    def forward_backward(
        self,
        emissions: torch.Tensor,
        log_pi: torch.Tensor,
        log_A: torch.Tensor,
        log_B: torch.Tensor,
    ):
        """
        Run forward-backward on one sequence.

        Returns:
            log_alpha: (T, K)
            log_beta:  (T, K)
            log_gamma: (T, K)   # state posteriors log P(z_t | x)
            log_xi:    (T-1, K, K)  # transition posteriors log P(z_t=i,z_{t+1}=j | x)
            log_likelihood: ()  # scalar
        """
        # 1. log_alpha = forward_algorithm(...)
        # 2. log_beta  = backward_algorithm(...)
        # 3. log_gamma from log_alpha + log_beta
        # 4. log_xi from α, β, log_A, log_B, emissions
        # 5. log_likelihood = logsumexp(log_alpha[-1])
        ...
    
    # ---- Main forward: compute loss on a batch of sentences ----

    def forward(self, batch_emissions: torch.Tensor) -> torch.Tensor:
        """
        batch_emissions: (B, T) padded integer word indices.

        Returns:
            loss: scalar = negative log-likelihood over batch.
        """
        # 1. Compute log_pi, log_A, log_B from neural nets once per batch
        # log_pi = self.compute_initial_log_probs()
        # log_A  = self.compute_transition_log_probs()
        # log_B  = self.compute_emission_log_probs()

        # 2. For each sequence in batch:
        #    - run forward algorithm (or forward_backward)
        #    - sum log-likelihoods
        # 3. Return negative average log-likelihood as loss
        ...
    
    # ---- Optional: function to get posteriors for analysis ----

    def infer_posteriors(self, emissions: torch.Tensor):
        """
        Convenience method to return γ and ξ for a single sequence,
        for analysis or EM-style updates if desired.
        """
        # log_pi, log_A, log_B = ...
        # return self.forward_backward(emissions, log_pi, log_A, log_B)
        ...



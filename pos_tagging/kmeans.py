from transformers import BertTokenizer, BertModel
from datasets import Dataset
import torch
import numpy as np
import tqdm

class KMeansPOSClusterer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()  # we’re just embedding, no training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.centroids = None


    def kmeans_lloyd(
            self,
            X: torch.Tensor,
            K: int,
            num_iters: int = 20,
            tol: float = 1e-4,
            verbose: bool = False
    ):
        N, D = X.shape
        # init centroids at random
        indices = torch.randperm(N, device=self.device)[:K]
        centroids = X[indices]
        prev_inertia = None
        for i in range(num_iters):
            dists = torch.cdist(X, centroids, p=2) ** 2
            labels = torch.argmin(dists,dim=1)
            inertia = dists[torch.arange(N, device=self.device), labels].sum()
            if prev_inertia is not None:
                rel_improvement = (prev_inertia - inertia).abs() / (prev_inertia + 1e-9)
                if rel_improvement < tol:
                    break
            prev_inertia = inertia
            centroids = torch.zeros(K, D, device=self.device, dtype=X.dtype)
            counts = torch.zeros(K, device=self.device,dtype=X.dtype)
            centroids.index_add_(0, labels, X)
            ones = torch.ones(N, device=self.device, dtype=X.dtype)
            counts.index_add(0, labels, ones)
            empty_mask = counts == 0
            non_empty_mask = ~empty_mask
            centroids[non_empty_mask] /= counts[non_empty_mask].unsqueeze(1)

            if empty_mask.any():
                n_empty = empty_mask.sum().item()
                rand_indices = torch.randperm(N, device=self.device)[:n_empty]
                centroids[empty_mask] = X[rand_indices]

        return centroids, labels

    def get_embeddings(self, inputs: Dataset) -> torch.Tensor:
        """
        inputs: HF Dataset with column 'form' = list of token strings per sentence
        returns: X [N_total_words, 768] on self.device
        """
        all_word_vecs = []

        for forms in tqdm.tqdm(inputs["form"], desc="Embedding sentences"):
            # forms: list of tokens -> we pass as list, not a single string
            w_embs = self.get_word_embeddings_for_sentence(forms)  # [T, 768]
            all_word_vecs.append(w_embs)

        X = torch.cat(all_word_vecs, dim=0).to(self.device)  # [N_total_words, 768]
        print(X.shape)
        return X




    def get_word_embeddings_for_sentence(self, tokens) -> torch.Tensor:
        """
        tokens: list of strings (word tokens) OR a raw string.
        returns [T, 768] tensor on self.device
        """
        # If you pass a list of tokens, use is_split_into_words=True
        if isinstance(tokens, list):
            enc = self.tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True
            )
        else:
            # fall back to raw string
            enc = self.tokenizer(
                tokens,
                return_tensors="pt",
                truncation=True
            )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)

        token_embeddings = outputs.last_hidden_state[0]  # [seq_len, 768]
        word_ids = enc.word_ids(batch_index=0)

        word_vecs = []
        buf = []
        current = None

        for tok_vec, w_id in zip(token_embeddings, word_ids):
            if w_id is None:  # special tokens
                continue
            if current is None:
                current = w_id

            if w_id != current:
                word_vecs.append(torch.stack(buf).mean(0))
                buf = []
                current = w_id

            buf.append(tok_vec)

        if buf:
            word_vecs.append(torch.stack(buf).mean(0))

        return torch.stack(word_vecs).to(self.device)  # [T, 768]




    
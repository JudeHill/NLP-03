from typing import Optional, Sequence
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
            counts.index_add_(0, labels, ones)
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
    
    @torch.no_grad()
    def predict_sentence(self, embs: torch.Tensor) -> torch.Tensor:
        """
        Assign each token embedding to the nearest centroid.

        Args:
            embs: [T, D] token embeddings (torch.Tensor). Can be on CPU or GPU.

        Returns:
            labels: [T] long tensor of cluster IDs on CPU (easy for eval/logging).
        """
        if self.centroids is None:
            raise ValueError("Centroids are not set. Call fit/train k-means first.")

        # Ensure tensor + float dtype
        if not isinstance(embs, torch.Tensor):
            embs = torch.tensor(embs, dtype=torch.float32)

        embs = embs.to(self.device, dtype=torch.float32)

        # Sanity check on dimensionality
        if embs.ndim != 2:
            raise ValueError(f"Expected embs with shape [T, D], got {tuple(embs.shape)}")
        if embs.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                f"Embedding dim mismatch: embs has D={embs.shape[1]}, "
                f"centroids have D={self.centroids.shape[1]}"
            )

        # Compute squared Euclidean distances: [T, K]
        # torch.cdist returns Euclidean distance; square it for squared distances
        dists = torch.cdist(embs, self.centroids, p=2) ** 2  # [T, K]

        # Nearest centroid per token
        labels = torch.argmin(dists, dim=1).to(torch.long)  # [T]

        return labels.cpu()
    
    @torch.no_grad()
    def embed_dataset(
        self,
        ds: Dataset,
        *,
        form_col: str = "form",
        out_col: str = "embeddings",
        truncate_other_token_cols: Optional[Sequence[str]] = ("tags",),
        show_progress: bool = True,
    ) -> Dataset:
        """
        Add per-sentence token embeddings to a HuggingFace Dataset.

        Args:
            ds: HF Dataset where each row has ds[form_col] = list[str] tokens.
            form_col: column name for tokenized sentence.
            out_col: output column name to store embeddings.
            truncate_other_token_cols: other per-token columns (e.g. "tags") to
                truncate if BERT truncation reduces the number of returned word vectors.
                Set to None or () to disable.
            show_progress: whether to show tqdm progress bar.

        Returns:
            New Dataset with an added column out_col.
            Each item in out_col is a nested Python list with shape [T, D].
        """
        if form_col not in ds.column_names:
            raise ValueError(f"Dataset is missing required column '{form_col}'")

        # We'll build new columns as Python lists then add them.
        all_embs = []
        new_cols: Dict[str, list] = {out_col: all_embs}

        # Prepare truncation buffers if requested and column exists
        trunc_cols = []
        if truncate_other_token_cols:
            for c in truncate_other_token_cols:
                if c in ds.column_names:
                    trunc_cols.append(c)
                    new_cols[c] = []

        iterator = ds if not show_progress else tqdm(ds, desc="Embedding dataset", total=len(ds))

        for ex in iterator:
            tokens = ex[form_col]  # list[str]

            # Compute [T', D] tensor on device
            w_embs = self.get_word_embeddings_for_sentence(tokens)  # torch.Tensor [T', D]
            T_prime = int(w_embs.shape[0])

            # Store as nested Python list so HF Dataset can hold it easily
            new_cols[out_col].append(w_embs.detach().cpu().tolist())

            # If BERT truncation reduced length, optionally truncate other token-level columns
            # to keep alignment: form, tags, etc.
            if trunc_cols:
                for c in trunc_cols:
                    seq = ex[c]
                    # Only truncate if it looks like a token-level sequence and lengths mismatch
                    if isinstance(seq, (list, tuple)) and len(seq) != T_prime:
                        new_cols[c].append(list(seq)[:T_prime])
                    else:
                        new_cols[c].append(seq)

        # Also consider truncating 'form' itself if mismatch (common when truncation happens)
        # We do it unconditionally if mismatch is detected.
        if "form" in ds.column_names:
            new_forms = []
            for ex, embs_list in zip(ds, new_cols[out_col]):
                T_prime = len(embs_list)
                forms = ex[form_col]
                if isinstance(forms, (list, tuple)) and len(forms) != T_prime:
                    new_forms.append(list(forms)[:T_prime])
                else:
                    new_forms.append(forms)
            new_cols[form_col] = new_forms

        # Add/replace columns
        ds2 = ds
        for c, values in new_cols.items():
            ds2 = ds2.remove_columns(c) if c in ds2.column_names else ds2
            ds2 = ds2.add_column(c, values)

        return ds2




    
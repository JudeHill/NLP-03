from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

def make_dataloader(hf_dataset, batch_size=32, pad_token_id=0):
    def collate_fn(batch):
        # batch is a list of dicts, each like {"input_ids": [...]}
        seqs = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)

        # pad to same length: (B, T_max)
        padded = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)

        return {
            "input_ids": padded,   # (B, T_max)
            "lengths": lengths,    # (B,)
        }

    loader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return loader

"""Protein language model embeddings: ESM-2 (general proteins) and AbLang2 (antibodies).

A protein language model reads a sequence and produces one vector per
residue. We average those vectors into a single fixed-length vector per
chain, which a simple regression can then use as features.
"""

from pathlib import Path

import numpy as np
import torch

ESM2_650M = "facebook/esm2_t33_650M_UR50D"


def _real_token_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """1 for residue tokens, 0 for the start token, end token and padding.

    ESM-2 adds a start token (BOS) before the sequence and an end token (EOS)
    after it, and batching pads shorter sequences. None of those are
    residues, so including them would shift the embedding, and padding would
    make a sequence's embedding depend on what else was in its batch.
    """
    mask = attention_mask.clone().float()
    mask[:, 0] = 0  # BOS is always the first token
    last_real = attention_mask.sum(dim=1) - 1  # EOS is the last real token
    mask[torch.arange(mask.shape[0]), last_real] = 0
    return mask


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average residue vectors, leaving out special and padding tokens."""
    mask = _real_token_mask(attention_mask)
    summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
    return summed / mask.sum(dim=1, keepdim=True)


def _strip(hidden: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    """Per-sequence residue vectors, with special and padding tokens removed."""
    mask = _real_token_mask(attention_mask).bool()
    return [hidden[i][mask[i]] for i in range(hidden.shape[0])]


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def esm2_embed(seqs: list[str], model_name: str = ESM2_650M, batch_size: int = 8) -> np.ndarray:
    """Mean-pooled last-layer ESM-2 embedding for each sequence."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = _device()
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    # Batch sequences of similar length together to minimize padding, then put
    # the results back in the original order.
    order = np.argsort([len(s) for s in seqs])
    pooled = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            idx = order[start : start + batch_size]
            batch = tokenizer([seqs[i] for i in idx], return_tensors="pt", padding=True).to(device)
            hidden = model(**batch).last_hidden_state
            pooled[idx] = mean_pool(hidden, batch["attention_mask"]).float().cpu().numpy()
    return pooled


def ablang2_embed(vhs: list[str], vls: list[str]) -> np.ndarray:
    """AbLang2 embedding of each heavy/light pair.

    AbLang2 was trained on paired antibody sequences, so it reads VH and VL
    together and returns one vector per antibody.
    """
    import ablang2

    model = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1, device="cpu")
    return np.asarray(model([[vh, vl] for vh, vl in zip(vhs, vls)], mode="seqcoding"), dtype=np.float32)


def save_cache(path: Path, names: list[str], X: np.ndarray) -> None:
    """Save embeddings together with the antibody names they belong to."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, names=np.array(names), X=X)


def load_cache(path: Path, names: list[str]) -> np.ndarray:
    """Load cached embeddings, refusing them if the antibody order changed.

    Rows are matched to antibodies by position, so a cache written for a
    different order would silently pair embeddings with the wrong labels.
    """
    stored = np.load(path)
    if list(stored["names"]) != list(names):
        raise ValueError(f"{path}: cached antibody names do not match the current data order")
    return stored["X"]


def esm2_per_residue(seqs: list[str], model_name: str = ESM2_650M, batch_size: int = 4) -> list[np.ndarray]:
    """One (len(seq), 1280) array per sequence: a vector for every residue.

    The GNN uses these as node features, so each residue carries what the
    language model inferred about it from the whole chain.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = _device()
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    order = np.argsort([len(s) for s in seqs])
    out: list[np.ndarray | None] = [None] * len(seqs)
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            idx = order[start : start + batch_size]
            batch = tokenizer([seqs[i] for i in idx], return_tensors="pt", padding=True).to(device)
            hidden = model(**batch).last_hidden_state
            for position, residues in zip(idx, _strip(hidden, batch["attention_mask"])):
                out[position] = residues.float().cpu().numpy()
    for seq, arr in zip(seqs, out):
        if arr is None or len(arr) != len(seq):
            raise ValueError("per-residue embedding length does not match its sequence")
    return out

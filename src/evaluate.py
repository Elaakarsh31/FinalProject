import math
import torch
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    top_k_accuracy_score,
)


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters, device, block_size=None):
    """
    Runs the model in eval mode over `eval_iters` batches for both splits
    and returns the mean loss for each. Takes the model as an explicit
    argument (rather than reading a global) so it can be reused across
    the baseline model, the tuned model, and every config in the grid
    search without them stepping on each other.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def collect_predictions(model, get_batch, device, n_batches=20, context_len=None, block_size=None):
    """
    Runs the model over several validation batches and collects, for every
    position in every sequence, the true next character and the model's
    predicted probability distribution over the vocabulary. Flattening
    this way turns "next-character prediction" into an ordinary multi-class
    classification problem: one row per character position, one label out
    of vocab_size classes.

    context_len, if given, truncates how many preceding characters the
    model is allowed to see before making each prediction (used for the
    context-length ablation in place of a traditional feature-importance
    analysis).
    """
    model.eval()
    all_probs = []
    all_targets = []

    for _ in range(n_batches):
        X, Y = get_batch("val")
        if context_len is not None and context_len < X.shape[1]:
            X = X[:, -context_len:]
            Y = Y[:, -context_len:]
        logits, _ = model(X)
        probs = F.softmax(logits, dim=-1)  # (B, T, vocab_size)
        B, T, C = probs.shape
        all_probs.append(probs.reshape(B * T, C).cpu())
        all_targets.append(Y.reshape(B * T).cpu())

    model.train()
    return torch.cat(all_probs, dim=0).numpy(), torch.cat(all_targets, dim=0).numpy()


def classification_report_dict(probs, targets, vocab_size):
    """
    Computes accuracy, top-5 accuracy, precision/recall/F1 (macro and
    weighted), and macro one-vs-rest ROC-AUC for the next-character
    prediction task, plus perplexity derived from cross-entropy.
    """
    preds = probs.argmax(axis=1)

    acc = accuracy_score(targets, preds)
    top5_acc = top_k_accuracy_score(targets, probs, k=5, labels=list(range(vocab_size)))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0
    )

    # cross-entropy / perplexity straight from the probabilities we already have
    eps = 1e-12
    true_probs = probs[range(len(targets)), targets]
    ce = -sum(math.log(max(p, eps)) for p in true_probs) / len(targets)
    perplexity = math.exp(ce)

    try:
        roc_auc_macro = roc_auc_score(
            targets, probs, multi_class="ovr", average="macro", labels=list(range(vocab_size))
        )
    except ValueError:
        # can happen if a class present in `labels` never appears in this sample
        roc_auc_macro = None

    return {
        "accuracy": acc,
        "top5_accuracy": top5_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "cross_entropy": ce,
        "perplexity": perplexity,
        "roc_auc_macro_ovr": roc_auc_macro,
    }


def top_confused_chars(probs, targets, itos, top_n=15):
    """
    Builds a confusion matrix restricted to the `top_n` most frequent
    characters in the validation targets, so the heatmap stays readable
    (full vocab_size x vocab_size for ~65 characters is mostly empty space).
    Returns the matrix plus the character labels used, in frequency order.
    """
    preds = probs.argmax(axis=1)
    counts = {}
    for t in targets:
        counts[t] = counts.get(t, 0) + 1
    top_ids = sorted(counts, key=counts.get, reverse=True)[:top_n]

    cm = confusion_matrix(targets, preds, labels=top_ids)
    labels = [repr(itos[i])[1:-1] for i in top_ids]  # e.g. '\n' -> \n, ' ' -> ' '
    return cm, labels

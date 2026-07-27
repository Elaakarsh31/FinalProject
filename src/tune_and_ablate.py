import itertools
import torch
from .model import GPTLanguageModel
from .train_utils import train_model
from .evaluate import estimate_loss, collect_predictions, classification_report_dict


def run_grid_search(param_grid, fixed_config, get_batch, device, search_iters, eval_interval, eval_iters):
    """
    Trains one short run per combination of hyperparameters in param_grid
    (e.g. n_layer, n_head, dropout, learning_rate) and records the final
    validation loss for each.

    param_grid: dict of {hyperparam_name: [values to try]}
    fixed_config: dict of hyperparams held constant across the grid
                  (must include vocab_size, block_size, n_embed unless
                  those are themselves in param_grid)
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    results = []

    for combo in combos:
        cfg = dict(fixed_config)
        cfg.update(dict(zip(keys, combo)))

        model = GPTLanguageModel(
            vocab_size=cfg["vocab_size"],
            block_size=cfg["block_size"],
            n_embed=cfg["n_embed"],
            n_head=cfg["n_head"],
            n_layer=cfg["n_layer"],
            dropout=cfg["dropout"],
        ).to(device)

        history = train_model(
            model, get_batch, device,
            max_iters=search_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            learning_rate=cfg["learning_rate"],
            verbose=False,
        )

        final_val_loss = history["val_loss"][-1]
        results.append({**dict(zip(keys, combo)), "val_loss": final_val_loss})
        print(f"config {dict(zip(keys, combo))} -> val loss {final_val_loss:.4f}")

    results.sort(key=lambda r: r["val_loss"])
    return results


def context_length_ablation(model, get_batch, device, vocab_size, context_lengths, n_batches=15):
    """
    Feature-importance analog for a sequence model
    """
    results = []
    for cl in context_lengths:
        probs, targets = collect_predictions(
            model, get_batch, device, n_batches=n_batches, context_len=cl
        )
        metrics = classification_report_dict(probs, targets, vocab_size)
        results.append({"context_length": cl, **metrics})
        print(f"context length {cl}: accuracy {metrics['accuracy']:.4f}, "
              f"perplexity {metrics['perplexity']:.4f}")
    return results

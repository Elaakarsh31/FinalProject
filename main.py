import os
import json
import torch

from src.dataloader import load_data, make_get_batch
from src.model import GPTLanguageModel
from src.bigram_model import BigramLanguageModel
from src.train_utils import train_model
from src.evaluate import collect_predictions, classification_report_dict, top_confused_chars
from src.tune_and_ablate import run_grid_search, context_length_ablation
from src.visualize import (
    plot_learning_curves, plot_confusion_matrix,
    plot_model_comparison, plot_hyperparam_results, plot_context_ablation,
)

DATA_PATH = "data/input.txt"          
RESULTS_DIR = "results"
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

BLOCK_SIZE = 256
BATCH_SIZE = 64
N_EMBED = 384
LEARNING_RATE = 3e-4

SEARCH_ITERS = 500     
FINAL_ITERS = 3000     
BIGRAM_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50

CONTEXT_LENGTHS = [1, 8, 32, 64, 128, 256]

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(1337)


def main():
    print(f"device: {device}")
    data = load_data(DATA_PATH, device)
    vocab_size = data["vocab_size"]
    get_batch = make_get_batch(data["train_data"], data["val_data"], BATCH_SIZE, BLOCK_SIZE, device)

    all_results = {}

    print("\n=== Training bigram baseline ===")
    bigram = BigramLanguageModel(vocab_size, N_EMBED, BLOCK_SIZE)
    bigram_history = train_model(
        bigram, get_batch, device,
        max_iters=BIGRAM_ITERS, eval_interval=EVAL_INTERVAL, eval_iters=EVAL_ITERS,
        learning_rate=LEARNING_RATE, label="bigram",
    )


    print("\n=== Hyperparameter grid search (GPT) ===")
    param_grid = {
        "n_layer": [4, 6],
        "n_head": [4, 6],
        "dropout": [0.1, 0.2],
    }
    fixed_config = {
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "n_embed": N_EMBED,
        "learning_rate": LEARNING_RATE,
    }
    grid_results = run_grid_search(
        param_grid, fixed_config, get_batch, device,
        search_iters=SEARCH_ITERS, eval_interval=EVAL_INTERVAL, eval_iters=EVAL_ITERS,
    )
    best = grid_results[0]
    print(f"\nBest config from search: {best}")

    print("\n=== Training final GPT model with best config ===")
    gpt = GPTLanguageModel(
        vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embed=N_EMBED,
        n_head=best["n_head"], n_layer=best["n_layer"], dropout=best["dropout"],
    )
    gpt_history = train_model(
        gpt, get_batch, device,
        max_iters=FINAL_ITERS, eval_interval=EVAL_INTERVAL, eval_iters=EVAL_ITERS,
        learning_rate=LEARNING_RATE, label="gpt-final",
    )
    torch.save(gpt.state_dict(), os.path.join(RESULTS_DIR, "gpt_final.pt"))


    print("\n=== Evaluating bigram ===")
    bigram_probs, bigram_targets = collect_predictions(bigram, get_batch, device, n_batches=20)
    bigram_metrics = classification_report_dict(bigram_probs, bigram_targets, vocab_size)
    print(bigram_metrics)

    print("\n=== Evaluating GPT ===")
    gpt_probs, gpt_targets = collect_predictions(gpt, get_batch, device, n_batches=20)
    gpt_metrics = classification_report_dict(gpt_probs, gpt_targets, vocab_size)
    print(gpt_metrics)

    cm, cm_labels = top_confused_chars(gpt_probs, gpt_targets, data["itos"], top_n=15)

    print("\n=== Context-length ablation ===")
    ablation_results = context_length_ablation(
        gpt, get_batch, device, vocab_size, CONTEXT_LENGTHS, n_batches=15
    )

    print("\n=== Positional embedding ablation ===")
    gpt_no_pos = GPTLanguageModel(
        vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embed=N_EMBED,
        n_head=best["n_head"], n_layer=best["n_layer"], dropout=best["dropout"],
        use_pos_embed=False,
    )
    no_pos_history = train_model(
        gpt_no_pos, get_batch, device,
        max_iters=FINAL_ITERS, eval_interval=EVAL_INTERVAL, eval_iters=EVAL_ITERS,
        learning_rate=LEARNING_RATE, label="gpt-no-posembed",
    )

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = data["decode"](gpt.generate(context, max_new_tokens=500)[0].tolist())
    with open(os.path.join(RESULTS_DIR, "sample_generation.txt"), "w") as f:
        f.write(sample)

    # -----------------------------------------------------------------
    # 8. Plots
    # -----------------------------------------------------------------
    plot_learning_curves(
        [bigram_history, gpt_history], ["Bigram", "GPT"],
        os.path.join(RESULTS_DIR, "learning_curves.png"),
    )
    plot_confusion_matrix(cm, cm_labels, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_hyperparam_results(grid_results, os.path.join(RESULTS_DIR, "hyperparam_search.png"))
    plot_context_ablation(ablation_results, os.path.join(RESULTS_DIR, "context_ablation.png"))

    metric_names = ["accuracy", "top5_accuracy", "f1_macro", "roc_auc_macro_ovr"]
    plot_model_comparison(
        metric_names, ["Bigram", "GPT"],
        [[bigram_metrics[m] for m in metric_names], [gpt_metrics[m] for m in metric_names]],
        os.path.join(RESULTS_DIR, "model_comparison.png"),
    )

    all_results = {
        "device": device,
        "bigram_metrics": bigram_metrics,
        "gpt_metrics": gpt_metrics,
        "grid_search_results": grid_results,
        "best_config": best,
        "context_ablation": ablation_results,
        "no_pos_embed_final_val_loss": no_pos_history["val_loss"][-1],
        "with_pos_embed_final_val_loss": gpt_history["val_loss"][-1],
        "bigram_history": bigram_history,
        "gpt_history": gpt_history,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. All results and plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

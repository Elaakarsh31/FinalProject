import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_learning_curves(histories, labels, out_path):
    """Train/val loss vs iteration for one or more models on the same axes."""
    plt.figure(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for i, (hist, label) in enumerate(zip(histories, labels)):
        c = colors[i % len(colors)]
        plt.plot(hist["iter"], hist["train_loss"], "--", color=c, label=f"{label} train")
        plt.plot(hist["iter"], hist["val_loss"], "-", color=c, label=f"{label} val")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy loss")
    plt.title("Learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, labels, out_path, title="Confusion matrix (top characters)"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted character")
    plt.ylabel("Actual character")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(metric_names, model_names, values, out_path, title="Model comparison"):
    """
    values: 2D list/array, shape (n_models, n_metrics)
    """
    values = np.array(values)
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)

    plt.figure(figsize=(8, 5))
    for i, model_name in enumerate(model_names):
        plt.bar(x + i * width, values[i], width, label=model_name)
    plt.xticks(x + width * (len(model_names) - 1) / 2, metric_names, rotation=20)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hyperparam_results(results, out_path, x_key="val_loss"):
    """Bar chart of validation loss for each configuration tried in the grid search."""
    labels = [", ".join(f"{k}={v}" for k, v in r.items() if k != "val_loss") for r in results]
    losses = [r["val_loss"] for r in results]

    plt.figure(figsize=(9, 5))
    plt.barh(labels, losses, color="steelblue")
    plt.xlabel("Final validation loss (lower is better)")
    plt.title("Hyperparameter grid search results")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_context_ablation(results, out_path):
    lengths = [r["context_length"] for r in results]
    acc = [r["accuracy"] for r in results]
    ppl = [r["perplexity"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lengths, acc, "o-", color="tab:blue", label="Accuracy")
    ax1.set_xlabel("Context length (characters of history given to the model)")
    ax1.set_ylabel("Next-character accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(lengths, ppl, "s--", color="tab:red", label="Perplexity")
    ax2.set_ylabel("Perplexity", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Effect of context length on prediction quality")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

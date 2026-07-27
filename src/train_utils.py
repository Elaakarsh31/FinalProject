import torch
from .evaluate import estimate_loss


def train_model(model, get_batch, device, max_iters, eval_interval, eval_iters,
                 learning_rate, verbose=True, label=""):

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = {"iter": [], "train_loss": [], "val_loss": []}

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, get_batch, eval_iters, device)
            history["iter"].append(it)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            if verbose:
                print(f"{label} iter {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return history

import matplotlib.pyplot as plt
from typing import List


def plot_history(train_history: List,
                 val_history: List,
                 metric: str,
                 title: str,
                 interval: int) -> None:
    """Plots the loss history"""
    epoch_idxs = range(1, len(train_history) + 1)
    plt.xticks(epoch_idxs[::interval], epoch_idxs[::interval])
    plt.plot(epoch_idxs, train_history, "-b", label="training")
    plt.plot(epoch_idxs, val_history, "-r", label="validation")
    plt.title(title)
    plt.legend()
    plt.ylabel(metric)
    plt.xlabel("Epochs")
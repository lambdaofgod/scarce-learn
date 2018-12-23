import matplotlib.pyplot as plt


def plot_training_metrics(train_accs, train_losses, n_epochs, n_episodes, epoch_ticks_interval=2):
    for name, metric in zip(["Accuracy", "Loss"], [train_accs, train_losses]):
        plt.plot(metric)
        plt.title(name)
        plt.xticks(range(0, n_epochs * n_episodes + 1, n_episodes * epoch_ticks_interval), range(0, n_epochs + 1, 5))
        plt.xlabel("epoch")
        plt.show()
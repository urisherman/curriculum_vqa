import matplotlib.pyplot as plt


def plot_training(train_loss, train_acc, dev_acc):

    fig, axs = plt.subplots(nrows=2, figsize=(15, 6))

    axs[0].plot(train_loss, '-', label='Train Loss');
    axs[0].legend()
    axs[1].plot(train_acc, '-o', label='Train Acc');
    axs[1].plot(dev_acc, '-o', label='Val Acc');
    axs[1].legend()

    plt.tight_layout()

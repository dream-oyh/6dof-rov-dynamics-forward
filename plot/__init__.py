import matplotlib.pyplot as plt


def plot(v_pred, v_true):
    label = ["u", "v", "w", "p", "q", "r"]
    plt.figure()
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(v_pred[:, i], label=f"{label[i]}_python")
        plt.plot(v_true[:, i], label=f"{label[i]}_matlab")
    plt.legend()
    plt.show()

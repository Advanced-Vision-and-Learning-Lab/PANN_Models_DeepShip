
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def load_confusion_matrices(filename):
    with open(filename, 'rb') as f:
        cm_list = pickle.load(f)
    return cm_list

def compute_metrics(cm_list):
    mean_cm = np.int64(np.ceil(np.mean(cm_list, axis=0)))
    std_cm = np.int64(np.ceil(np.std(cm_list, axis=0)))
    total = np.sum(mean_cm, axis=1, keepdims=True)
    accuracies = mean_cm.astype('float') / total
    std_dev_accuracies = std_cm.astype('float') / (total + 10e-6)  
    return mean_cm, std_cm, accuracies * 100, std_dev_accuracies * 100

def plot_confusion_matrix(cm, std_cm, accuracies, std_dev_accuracies, classes,
                          title='Confusion Matrix', cmap=plt.cm.Blues, fontsize=14, show_percent=True):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    if show_percent:
        cm_percent = accuracies
        cm_percent_std = std_dev_accuracies
        im = ax.imshow(cm_percent, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=fontsize)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=fontsize)
    ax.set_yticklabels(classes, fontsize=fontsize)
    ax.set_ylim(len(classes)-0.5, -0.5)

    fmt = '.2f' if show_percent else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if show_percent:
            entry_text = str(format(cm[i, j], fmt) + '±' + format(std_cm[i, j], fmt))
            accuracy_text = '(' + str(format(cm_percent[i, j], fmt)) + '±' + str(format(cm_percent_std[i, j], fmt)) + '%)'
            text = entry_text + '\n' + accuracy_text
        else:
            text = str(format(cm[i, j], fmt) + '±' + format(std_cm[i, j], fmt))

        ax.text(j, i, text, ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('tb_logs/Figures/confusion_matrix_STFT_b64_2.png')






cm_list = load_confusion_matrices('tb_logs/CM_Matrices/confusion_matrices_STFT_b64.pkl')
mean_cm, std_cm, accuracies, std_dev_accuracies = compute_metrics(cm_list)
class_names = ['Cargo', 'Passengership', 'Tanker', 'Tug']
plot_confusion_matrix(mean_cm, std_cm, accuracies, std_dev_accuracies, class_names, show_percent=True)


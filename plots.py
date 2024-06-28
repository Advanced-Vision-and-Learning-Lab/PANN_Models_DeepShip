import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_features(file_path):
    data = torch.load(file_path)
    features = data['features']
    labels = data['labels']
    preds = data.get('preds')  # Make sure to load predictions as well
    return features, labels, preds

def compute_tsne(features):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)
    return tsne_results

def plot_tsne(tsne_results, labels, class_names, title):
    plt.figure(figsize=(10, 10), dpi=150)  # Increase size and resolution
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=6)
    
    # Create a legend with class names
    handles, _ = scatter.legend_elements()
    legend_labels = [class_names[int(label)] for label in torch.unique(labels)]
    plt.legend(handles, legend_labels, fontsize=14)  # Increase font size of legend labels
    
    plt.title(title, fontsize=18)
    plt.xlabel("TSNE Component 1", fontsize=16)  # Rename axes for better clarity
    plt.ylabel("TSNE Component 2", fontsize=16)
    plt.xticks(fontsize=14)  # Increase font size of x-axis values
    plt.yticks(fontsize=14)  # Increase font size of y-axis values
    plt.tight_layout()
    plt.savefig(f"features/{title.replace(' ', '_')}.png")
    plt.close()


def plot_confusion_matrix(preds, labels, class_names, output_file):
    # Convert predictions to class labels if they are logits
    if preds.ndim > 1 and preds.size(1) > 1:
        preds = preds.argmax(dim=1)
    
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))  # Make the plot bigger

    disp.plot(cmap='Blues', ax=ax, colorbar=False)  # Disable default colorbar

    # Adjust the font size of annotations
    for text in disp.text_.ravel():
        text.set_fontsize(18)
        
    # Add a colorbar with correct size and position
    cbar = fig.colorbar(disp.im_, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)  # Increase the colorbar font size
    
    # Set font sizes for labels and titles
    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.title('Confusion Matrix', fontsize=22)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()



def generate_tsne_plots(model_name, class_names):
    # Load test features, labels, and predictions
    test_features, test_labels, test_preds = load_features(f"features/{model_name}_test_features.pth")
    test_tsne_results = compute_tsne(test_features)
    plot_tsne(test_tsne_results, test_labels, class_names, title=f"{model_name} Test t-SNE Plot")

if __name__ == "__main__":
    class_names = ['Cargo', 'Passengership', 'Tanker', 'Tug']
    model_name = "CNN_14_16k"

    #generate_tsne_plots(model_name, class_names)

    # Plot confusion matrix
    _, test_labels, test_preds = load_features(f"features/{model_name}_test_features.pth")
    plot_confusion_matrix(test_preds, test_labels, class_names, output_file=f"features/{model_name}_confusion_matrix_test.png")

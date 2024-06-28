import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_features(file_path):
    data = torch.load(file_path)
    features = data['features']
    labels = data['labels']
    return features, labels

def compute_tsne(features):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)
    return tsne_results

def plot_tsne(tsne_results, labels, title):
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=5)
    plt.legend(handles=scatter.legend_elements()[0], labels=set(labels))
    plt.title(title)
    plt.savefig(f"features/{title.replace(' ', '_')}.png")
    plt.close()

def plot_confusion_matrix(preds, labels, class_names, output_file):
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='viridis')
    plt.savefig(output_file)
    plt.close()

def generate_tsne_plots():
    # Load test features and labels
    test_features, test_labels = load_features("features/test_features.pth")
    test_tsne_results = compute_tsne(test_features)
    plot_tsne(test_tsne_results, test_labels, title="Test t-SNE Plot")

if __name__ == "__main__":
    generate_tsne_plots()
    class_names = ['Cargo', 'Passengership', 'Tanker', 'Tug']
    
    # Load test predictions and labels
    test_preds = torch.load("features/test_preds.pth")
    test_labels = torch.load("features/test_labels.pth")
    
    plot_confusion_matrix(test_preds, test_labels, class_names, output_file="features/confusion_matrix_test.png")

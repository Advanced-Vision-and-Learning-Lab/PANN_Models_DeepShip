import pdb

import torch
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Utils.LitModel import LitModel
from Datasets.SSDataModule import SSAudioDataModule
from Demo_Parameters import Parameters

# Create a mock args class to simulate argparse
class MockArgs:
    def __init__(self):
        self.save_results = True
        self.folder = 'Saved_Models/'
        self.model = 'CNN_14_32k'  
        self.histogram = False
        self.data_selection = 0
        self.numBins = 16
        self.feature_extraction = False
        self.use_pretrained = True
        self.train_batch_size = 64
        self.val_batch_size = 128
        self.test_batch_size = 128
        self.num_epochs = 1
        self.resize_size = 256
        self.lr = 5e-5
        self.use_cuda = True
        self.audio_feature = 'STFT'
        self.optimizer = 'Adam'
        self.patience = 1
        self.sample_rate = 32000

# Instantiate mock args and load parameters
args = MockArgs()
Params = Parameters(args)

# Set up constants from Params dictionary
s_rate = Params['sample_rate']
Dataset_n = Params['Dataset']
model_name = Params['Model_name']
num_classes = Params['num_classes'][Dataset_n]

batch_size = Params['batch_size']['train']
data_dir = Params["data_dir"]
new_dir = Params["new_dir"]

# Set up CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set PyTorch precision
torch.set_float32_matmul_precision('medium')
data_dir = './Datasets/DeepShip' 
batch_size = 32
sample_rate = 32000

data_module = SSAudioDataModule(new_dir, batch_size=batch_size, sample_rate=Params['sample_rate'])
data_module.prepare_data()

split_indices_path = 'split_indices.txt' 

# Create a mapping of classes to indices 
class_to_idx = {
    'Cargo': 0,
    'Passengership': 1,
    'Tanker': 2,
    'Tug': 3,
}

# Define the run number and model checkpoint path
run_number = 0  # Change this as needed to select Run_0, Run_1, or Run_2
model_folder = f'PANN_Weights/CNN_14_32k_b32_32000/Run_{run_number}/CNN_14_32k/version_0/checkpoints/'

# Automatically find the checkpoint file in the directory
checkpoint_files = [f for f in os.listdir(model_folder) if f.endswith('.ckpt')]
best_model_path = os.path.join(model_folder, checkpoint_files[0])  # Use the first found checkpoint

# Load the best model from checkpoint
best_model = LitModel.load_from_checkpoint(
    checkpoint_path=best_model_path,
    Params=Params,
    model_name=model_name,
    num_classes=num_classes,
    Dataset=Dataset_n,
    pretrained_loaded=True,
    run_number=run_number
)

# Move the model to the appropriate device (GPU or CPU)
best_model.to(device)

# Create a test dataloader
test_loader = data_module.test_dataloader()

# Print model structure for reference
print("Model Architecture:\n", best_model)

# Evaluate test accuracy
best_model.eval()  # Set the model to evaluation mode

print('\n')

# Select one correctly classified sample per class for CAM analysis
correct_samples_per_class = {}

# Add a flag to ensure we only print once
printed_once = False

for batch in test_loader:
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    
    # Forward pass through the model
    outputs = best_model(inputs)

    # Print shapes for debugging only once
    if not printed_once:
        print("First element of outputs:", outputs[0].shape)
        print("Second element of outputs (logits):", outputs[1].shape)
        printed_once = True

    # Extract logits and compute predictions
    logits = outputs[1]
    _, preds = torch.max(logits, dim=1)

    # Check for correctly classified samples within valid index range
    for i in range(len(labels)):  
        if preds[i] == labels[i]:
            class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(labels[i].item())]
            if class_name not in correct_samples_per_class:
                correct_samples_per_class[class_name] = (inputs[i], labels[i])
        
        # Stop once we have one sample per class
        if len(correct_samples_per_class) == len(class_to_idx):
            break

    # Break outer loop if we have all classes covered
    if len(correct_samples_per_class) == len(class_to_idx):
        break

### CAM ###
from contextlib import contextmanager
@contextmanager
def register_hooks(layer, forward_hook, backward_hook):
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)
    try:
        yield
    finally:
        forward_handle.remove()
        backward_handle.remove()

def generate_gradcam(model, input_tensor, target_class, last_conv_layer):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    with register_hooks(last_conv_layer, forward_hook, backward_hook):
        # Forward pass
        model.eval()
        logits = model(input_tensor)[1]

        # Backward pass for target class
        model.zero_grad()
        target_score = logits[0][target_class]
        target_score.backward()

    # Get gradients and activations
    gradients = gradients[0]
    activations = activations[0]

    # Compute weights using GAP
    weights = torch.mean(gradients, dim=(2, 3))

    # Compute Grad-CAM
    cam = torch.zeros(activations.shape[2:], device=input_tensor.device)
    for i in range(weights.shape[1]):
        cam += weights[0, i] * activations[0, i]

    cam = F.relu(cam)  # Apply ReLU
    # cam -= cam.min()
    # if cam.max() > 0:
    #     cam /= cam.max()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    return cam.cpu().detach().numpy()
    
    
    
# Dictionaries to store correctly and misclassified samples per class
correct_samples_per_class = {class_name: [] for class_name in class_to_idx.keys()}
misclassified_samples_per_class = {class_name: [] for class_name in class_to_idx.keys()}

# Populate dictionaries with correctly and misclassified samples
for batch in test_loader:
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = best_model(inputs)
    _, preds = torch.max(outputs[1], dim=1)  # Get predictions

    for i in range(len(labels)):
        true_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(labels[i].item())]
        predicted_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(preds[i].item())]

        if preds[i] == labels[i]:  # Correctly classified
            correct_samples_per_class[true_class_name].append((inputs[i], labels[i]))
        else:  # Misclassified
            misclassified_samples_per_class[true_class_name].append((inputs[i], preds[i]))

last_conv_layer = best_model.model_ft.backbone.conv_block6

# Function to process Grad-CAM for a given set of samples (correct or misclassified)
def process_gradcam(samples_dict, save_path_prefix, description):
    save_single_example = True  # Set this to True to save a single example per class
    for class_name, samples in samples_dict.items():
        print(f"Processing Grad-CAM for {len(samples)} {description} samples in class: {class_name}")

        aggregated_cam = None
        single_example_saved = False  # Track if a single example has been saved

        for idx, (sample_input, target_label) in enumerate(samples):
            sample_input = sample_input.unsqueeze(0).to(device)  # Add batch dimension
            target_class = target_label.item() if description == "correctly classified" else target_label.item()

            # Compute log-mel spectrogram for current sample
            with torch.no_grad():
                spectrogram_output = best_model.model_ft.backbone.spectrogram_extractor(sample_input)
                logmel_output = best_model.model_ft.backbone.logmel_extractor(spectrogram_output)

            # Generate Grad-CAM heatmap
            cam = generate_gradcam(best_model, sample_input, target_class, last_conv_layer)

            # Resize CAM to match input dimensions (e.g., spectrogram size)
            cam_resized = F.interpolate(torch.tensor(cam).unsqueeze(0).unsqueeze(0), size=(501, 64), mode='bilinear', align_corners=False)
            cam_resized_np = cam_resized.squeeze().numpy()
            
            # Normalize the individual CAM before aggregation
            cam_resized_np = (cam_resized_np - cam_resized_np.min()) / (cam_resized_np.max() + 1e-8)
    
            # Aggregate CAMs (e.g., sum or average)
            if aggregated_cam is None:
                aggregated_cam = cam_resized_np
            else:
                aggregated_cam += cam_resized_np

            # Save a single example if requested and not already saved
            if save_single_example and not single_example_saved:
                logmel_output_np = logmel_output.squeeze(0).squeeze(0).cpu().numpy()  # Convert log-mel spectrogram to NumPy array

                # Save the single example Grad-CAM with original spectrogram as subplot
                plt.figure(figsize=(15, 5))

                # Subplot 1: Original Log-Mel Spectrogram
                plt.subplot(1, 2, 1)
                plt.title(f"Log-Mel Spectrogram ({class_name}, Single Example)")
                plt.imshow(logmel_output_np, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar()

                # Subplot 2: Grad-CAM Heatmap Overlayed on Spectrogram
                plt.subplot(1, 2, 2)
                plt.title(f"Grad-CAM Heatmap ({class_name}, Single Example)")
                plt.imshow(logmel_output_np, aspect='auto', origin='lower', cmap='viridis')  # Background spectrogram
                plt.imshow(cam_resized_np, aspect='auto', origin='lower', cmap='jet', alpha=0.5)  # Overlay CAM with transparency
                plt.colorbar()

                plt.savefig(f"{save_path_prefix}_{class_name}_single.png", dpi=300)
                plt.close()

                single_example_saved = True  # Mark that the single example has been saved

        # Average CAM across all samples
        # Average CAM across all samples and normalize
        if len(samples) > 0:
            aggregated_cam /= len(samples)
            aggregated_cam = (aggregated_cam - aggregated_cam.min()) / (aggregated_cam.max() + 1e-8)

        # Save aggregated CAM visualization
        plt.figure(figsize=(8, 6))
        plt.title(f"Aggregated Grad-CAM Heatmap ({class_name}, {description})")
        plt.imshow(logmel_output.squeeze(0).squeeze(0).cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.imshow(aggregated_cam, aspect='auto', origin='lower', cmap='jet', alpha=0.5, vmin=0, vmax=1)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_{class_name}_{description}_aggregated.png", dpi=300)
        plt.close()

# Process Grad-CAM for correctly classified samples
process_gradcam(correct_samples_per_class, "cam/figures_pann/gradcam", "correctly classified")

# Process Grad-CAM for misclassified samples
process_gradcam(misclassified_samples_per_class, "cam/figures_pann/gradcam", "misclassified")    



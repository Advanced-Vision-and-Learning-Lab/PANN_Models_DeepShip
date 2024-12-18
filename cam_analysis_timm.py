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
        self.model = 'convnextv2_tiny.fcmae' 
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

split_indices_path = 'split_indices.txt'  # Path to split indices file

# Create a mapping of classes to indices
class_to_idx = {
    'Cargo': 0,
    'Passengership': 1,
    'Tanker': 2,
    'Tug': 3,
}

# Define the run number and model checkpoint path
run_number = 0  # Change this as needed to select Run_0, Run_1, or Run_2
model_folder = f'PANN_Weights/convnextv2_tiny.fcmae_b64_32000/Run_{run_number}/convnextv2_tiny.fcmae/version_0/checkpoints/'

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
# from contextlib import contextmanager
# @contextmanager
# def register_hooks(layer, forward_hook, backward_hook):
#     forward_handle = layer.register_forward_hook(forward_hook)
#     backward_handle = layer.register_full_backward_hook(backward_hook)
#     try:
#         yield
#     finally:
#         forward_handle.remove()
#         backward_handle.remove()

# def generate_gradcam(model, input_tensor, target_class, last_conv_layer):
#     gradients = []
#     activations = []

#     def backward_hook(module, grad_input, grad_output):
#         gradients.append(grad_output[0])

#     def forward_hook(module, input, output):
#         activations.append(output)

#     with register_hooks(last_conv_layer, forward_hook, backward_hook):
#         # Forward pass
#         model.eval()
#         logits = model(input_tensor)[1]  # Get logits from fully connected layer

#         # Backward pass for target class
#         model.zero_grad()
#         target_score = logits[0][target_class]
#         target_score.backward()

#     # Get gradients and activations
#     gradients = gradients[0]
#     activations = activations[0]

#     # Compute weights using GAP
#     weights = torch.mean(gradients, dim=(2, 3))

#     # Compute Grad-CAM
#     cam = torch.zeros(activations.shape[2:], device=input_tensor.device)
#     for i in range(weights.shape[1]):
#         cam += weights[0, i] * activations[0, i]

#     cam = F.relu(cam)  # Apply ReLU
#     # cam -= cam.min()
#     # if cam.max() > 0:
#     #     cam /= cam.max()
#     cam = (cam - cam.min()) / (cam.max() + 1e-8)

#     return cam.cpu().detach().numpy()


# # Dictionaries to store correctly and misclassified samples per class
# correct_samples_per_class = {class_name: [] for class_name in class_to_idx.keys()}
# misclassified_samples_per_class = {class_name: [] for class_name in class_to_idx.keys()}

# # Populate dictionaries with correctly and misclassified samples
# for batch in test_loader:
#     inputs, labels = batch
#     inputs, labels = inputs.to(device), labels.to(device)
    
#     # Pass raw waveform directly to the model
#     outputs = best_model(inputs)  # The model handles spectrogram/log-mel extraction internally
#     _, preds = torch.max(outputs[1], dim=1)  # Get predictions

#     for i in range(len(labels)):
#         true_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(labels[i].item())]
        
#         if preds[i] == labels[i]:  # Correctly classified
#             correct_samples_per_class[true_class_name].append((inputs[i], labels[i]))
#         else:  # Misclassified
#             predicted_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(preds[i].item())]
#             misclassified_samples_per_class[true_class_name].append((inputs[i], labels[i], predicted_class_name))

# # Adjust last convolutional layer for ConvNeXt-based model
# last_conv_layer = best_model.model_ft.backbone.stages[-1].blocks[-1].conv_dw  # Last depthwise conv layer

# # Function to process Grad-CAM for a set of samples (correct or misclassified)
# def process_gradcam(samples_dict, description, save_single_example=True):
#     for class_name, samples in samples_dict.items():
#         print(f"Processing Grad-CAM for {len(samples)} {description} samples in class: {class_name}")

#         aggregated_cam = None
#         single_example_saved = False  # Track if a single example has been saved

#         for idx, sample_data in enumerate(samples):
#             sample_input = sample_data[0].unsqueeze(0).to(device)  # Add batch dimension
#             target_class = sample_data[1].item()  # Get target class index

#             # Pass raw waveform directly to the model (no external spectrogram extraction)
#             outputs = best_model(sample_input)  # The model handles spectrogram/log-mel extraction internally

#             # Generate Grad-CAM heatmap using the last convolutional layer
#             cam = generate_gradcam(best_model, sample_input, target_class, last_conv_layer)

#             # Resize CAM to match input dimensions (e.g., spectrogram size)
#             cam_resized = F.interpolate(torch.tensor(cam).unsqueeze(0).unsqueeze(0), size=(501, 64), mode='bilinear', align_corners=False)
#             cam_resized_np = cam_resized.squeeze().numpy()

#             # Normalize the individual CAM before aggregation
#             cam_resized_np = (cam_resized_np - cam_resized_np.min()) / (cam_resized_np.max() + 1e-8)
    
#             # Aggregate CAMs (e.g., sum or average)
#             if aggregated_cam is None:
#                 aggregated_cam = cam_resized_np
#             else:
#                 aggregated_cam += cam_resized_np

#             # Save a single example if requested and not already saved
#             if save_single_example and not single_example_saved:
#                 with torch.no_grad():
#                     spectrogram_output = best_model.mel_extractor.spectrogram_extractor(sample_input)
#                     logmel_output = best_model.mel_extractor.logmel_extractor(spectrogram_output)

#                 logmel_output_np = logmel_output.squeeze(0).squeeze(0).cpu().numpy()  # Convert log-mel spectrogram to NumPy array

#                 plt.figure(figsize=(15, 5))

#                 # Subplot 1: Original Log-Mel Spectrogram
#                 plt.subplot(1, 2, 1)
#                 plt.title(f"Log-Mel Spectrogram ({class_name}, Single Example)")
#                 plt.imshow(logmel_output_np, aspect='auto', origin='lower', cmap='viridis')
#                 plt.colorbar()

#                 # Subplot 2: Grad-CAM Heatmap Overlayed on Spectrogram
#                 plt.subplot(1, 2, 2)
#                 plt.title(f"Grad-CAM Heatmap ({class_name}, Single Example)")
#                 plt.imshow(logmel_output_np, aspect='auto', origin='lower', cmap='viridis')  # Background spectrogram
#                 plt.imshow(cam_resized_np, aspect='auto', origin='lower', cmap='jet', alpha=0.5)  # Overlay CAM with transparency
#                 plt.colorbar()

#                 output_path = f"cam/figures_timm/gradcam_{class_name}_{description}_single.png"
#                 plt.savefig(output_path, dpi=300)
#                 plt.close()

#                 single_example_saved = True  # Mark that the single example has been saved

#         if len(samples) > 0:
#             aggregated_cam /= len(samples)
#             aggregated_cam = (aggregated_cam - aggregated_cam.min()) / (aggregated_cam.max() + 1e-8)

#         # Save aggregated CAM visualization
#         plt.figure(figsize=(8, 6))
#         plt.title(f"Aggregated Grad-CAM Heatmap ({class_name}, {description})")
#         plt.imshow(aggregated_cam, aspect='auto', origin='lower', cmap='jet', alpha=0.5, vmin=0, vmax=1)
#         plt.colorbar()
#         plt.tight_layout()
#         plt.savefig(f"cam/figures_timm/gradcam_{class_name}_{description}_aggregated.png", dpi=300)
#         plt.close()


# # Process Grad-CAM for correctly classified samples
# process_gradcam(correct_samples_per_class, "correctly classified")

# # Process Grad-CAM for misclassified samples
# process_gradcam(misclassified_samples_per_class, "misclassified")



### CAM ###
import torch
import os
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
        logits = model(input_tensor)[1]  # Get logits from fully connected layer

        # Backward pass for target class
        model.zero_grad()
        target_score = logits[0][target_class]
        target_score.backward()

    # Get gradients and activations
    gradients = gradients[0]
    activations = activations[0]

    # Compute weights using Global Average Pooling (GAP)
    weights = torch.mean(gradients, dim=(2, 3))

    # Compute Grad-CAM
    cam = torch.zeros(activations.shape[2:], device=input_tensor.device)
    for i in range(weights.shape[1]):
        cam += weights[0, i] * activations[0, i]

    cam = F.relu(cam)  # Apply ReLU
    cam = (cam - cam.min()) / (cam.max() + 1e-8)  # Normalize

    return cam.cpu().detach().numpy()

# Dictionaries to store correctly and misclassified samples per class
correct_samples_per_class = {class_name: [] for class_name in class_to_idx.keys()}
misclassified_samples_per_class = {class_name: [] for class_name in class_to_idx.keys()}

# Populate dictionaries with correctly and misclassified samples
for batch in test_loader:
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    
    # Pass raw waveform directly to the model
    outputs = best_model(inputs)  # The model handles spectrogram/log-mel extraction internally
    _, preds = torch.max(outputs[1], dim=1)  # Get predictions

    for i in range(len(labels)):
        true_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(labels[i].item())]
        
        if preds[i] == labels[i]:  # Correctly classified
            correct_samples_per_class[true_class_name].append((inputs[i], labels[i]))
        else:  # Misclassified
            predicted_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(preds[i].item())]
            misclassified_samples_per_class[true_class_name].append((inputs[i], labels[i], predicted_class_name))

# Adjust last convolutional layer for ConvNeXt-based model
last_conv_layer = best_model.model_ft.backbone.stages[-1].blocks[-1].conv_dw  # Last depthwise conv layer

# Save directory
save_directory = "cam/figures_timm"
os.makedirs(save_directory, exist_ok=True)

# Function to process Grad-CAM for a set of samples (correct or misclassified)
def process_gradcam(samples_dict, description, save_single_example=True):
    for class_name, samples in samples_dict.items():
        print(f"Processing Grad-CAM for {len(samples)} {description} samples in class: {class_name}")

        aggregated_cam = None
        single_example_saved = False  # Track if a single example has been saved

        for idx, sample_data in enumerate(samples):
            sample_input = sample_data[0].unsqueeze(0).to(device)  # Add batch dimension
            target_class = sample_data[1].item()  # Get target class index

            # Pass raw waveform directly to the model (no external spectrogram extraction)
            outputs = best_model(sample_input)  # The model handles spectrogram/log-mel extraction internally

            # Generate Grad-CAM heatmap using the last convolutional layer
            cam = generate_gradcam(best_model, sample_input, target_class, last_conv_layer)

            # Resize CAM to match input dimensions (e.g., spectrogram size)
            cam_resized = F.interpolate(torch.tensor(cam).unsqueeze(0).unsqueeze(0), size=(501, 64), mode='bilinear', align_corners=False)
            cam_resized_np = cam_resized.squeeze().numpy()

            # Normalize the individual CAM before aggregation
            cam_resized_np = (cam_resized_np - cam_resized_np.min()) / (cam_resized_np.max() + 1e-8)

            # Aggregate CAMs (e.g., sum)
            if aggregated_cam is None:
                aggregated_cam = cam_resized_np
            else:
                aggregated_cam += cam_resized_np

            # Save a single example if requested and not already saved
            if save_single_example and not single_example_saved:
                with torch.no_grad():
                    # Assuming the model has a method to extract log-mel spectrogram
                    spectrogram_output = best_model.mel_extractor.spectrogram_extractor(sample_input)
                    logmel_output = best_model.mel_extractor.logmel_extractor(spectrogram_output)

                logmel_output_np = logmel_output.squeeze(0).squeeze(0).cpu().numpy()  # Convert to NumPy array

                plt.figure(figsize=(15, 15)) 

                # Subplot 1: Original Log-Mel Spectrogram
                plt.subplot(1, 2, 1)
                plt.imshow(logmel_output_np, aspect='auto', origin='lower', cmap='viridis')
                plt.title(f"Log-Mel Spectrogram ({class_name}, Single Example)", fontsize=16)
                plt.xlabel('Time (s)', fontsize=14)         
                plt.ylabel('Frequency (Hz)', fontsize=14)


                # Subplot 2: Grad-CAM Heatmap Overlayed on Spectrogram
                plt.subplot(1, 2, 2)
                plt.imshow(logmel_output_np, aspect='auto', origin='lower', cmap='viridis')  # Background spectrogram
                plt.imshow(cam_resized_np, aspect='auto', origin='lower', cmap='jet', alpha=0.5)  # Overlay CAM with transparency
                plt.title(f"Grad-CAM Heatmap ({class_name}, Single Example)")
                plt.xlabel('Time (s)')         
                plt.ylabel('Frequency (Hz)')


                # Save the single example figure with annotations
                output_path_single = os.path.join(save_directory, f"gradcam_{class_name}_{description}_single.png")
                plt.tight_layout()
                plt.savefig(output_path_single, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

                single_example_saved = True  # Mark that the single example has been saved

        if len(samples) > 0:
            aggregated_cam /= len(samples)
            aggregated_cam = (aggregated_cam - aggregated_cam.min()) / (aggregated_cam.max() + 1e-8)

        # Save aggregated CAM with labels, titles, and colorbars
        plt.figure(figsize=(8, 6))  # Adjust as needed
        plt.title(f"Aggregated Grad-CAM Heatmap ({class_name}, {description})")
        plt.imshow(aggregated_cam, aspect='auto', origin='lower', cmap='jet', alpha=0.5, vmin=0, vmax=1)
        plt.colorbar()
        plt.tight_layout()
        output_path_aggregated = os.path.join(save_directory, f"gradcam_{class_name}_{description}_aggregated.png")
        plt.savefig(output_path_aggregated, dpi=300)
        plt.close()

        # Save aggregated CAM without labels, titles, axes, or colorbars
        plt.figure(figsize=(3, 6), dpi=600)  # Longer on y-axis, high resolution
        plt.imshow(aggregated_cam, aspect='auto', origin='lower', cmap='jet', alpha=0.5, vmin=0, vmax=1)
        plt.axis('off')  # Remove axes
        plt.tight_layout(pad=0)
        output_path_aggregated_no_labels = os.path.join(save_directory, f"gradcam_{class_name}_{description}_aggregated_no_labels.png")
        plt.savefig(output_path_aggregated_no_labels, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()

# Function to save a single colorbar
def save_colorbar(save_path, cmap_name='jet', orientation='vertical'):
    fig, ax = plt.subplots(figsize=(1, 6))  # Adjust width and height as needed
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create colorbar without any labels or ticks
    cbar = plt.colorbar(sm, cax=ax, orientation=orientation, ticks=[])
    cbar.outline.set_visible(False)  # Remove the outline

    ax.axis('off')  # Remove axis
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

# Example usage: Process Grad-CAM for correctly classified and misclassified samples
process_gradcam(correct_samples_per_class, "correctly classified")
process_gradcam(misclassified_samples_per_class, "misclassified")

# Save the colorbar once after processing all classes
colorbar_path = os.path.join(save_directory, "colorbar.png")
save_colorbar(colorbar_path)

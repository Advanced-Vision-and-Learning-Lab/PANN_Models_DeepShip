import pdb

import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from Utils.LitModel import LitModel
from Datasets.SSDataModule import SSAudioDataModule
from Demo_Parameters import Parameters
import os
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile  

# Create a mock args class to simulate argparse
class MockArgs:
    def __init__(self):
        self.save_results = True
        self.folder = 'Saved_Models/'
        self.model = 'CNN_14_32k'  # or 'convnextv2_tiny.fcmae'
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
data_dir = './Datasets/DeepShip'  # Adjust this path as needed
batch_size = 32
sample_rate = 32000

data_module = SSAudioDataModule(new_dir, batch_size=batch_size, sample_rate=Params['sample_rate'])
data_module.prepare_data()

# Usage Example: Create a DataLoader for Grad-CAM analysis
split_indices_path = 'split_indices.txt'  # Path to your split indices file

# Create a mapping of classes to indices (you should have this defined elsewhere)
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
correct = 0
total = 0

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
        print("First element of outputs (intermediate result):", outputs[0].shape)
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


# Print selected samples per class
for class_name, (input_sample, label) in correct_samples_per_class.items():
    print(f"Selected Sample for {class_name}:")
    print(f"Input Tensor Shape: {input_sample.shape}")
    print(f"Label: {label.item()}")

# Print spectrogram layer output for one sample (e.g., Cargo)
# Print spectrogram layer output for one sample (e.g., Cargo)
sample_input, _ = correct_samples_per_class['Cargo']
with torch.no_grad():
    # Pass through spectrogram extractor and logmel extractor
    spectrogram_output = best_model.model_ft.backbone.spectrogram_extractor(sample_input.unsqueeze(0))
    logmel_output = best_model.model_ft.backbone.logmel_extractor(spectrogram_output)

# Print shapes for debugging
print("Spectrogram Output Shape:", spectrogram_output.shape)  # This will still be [1, 1, 501, 513]
print("Log-Mel Spectrogram Output Shape:", logmel_output.shape)  # This should now be [1, 64, 501]


# Choose a layer for CAM - typically a deeper convolutional layer provides more meaningful activations.
chosen_layer = best_model.model_ft.backbone.conv_block5.conv2  # Example: Conv2d(512, 1024)

print("Chosen Layer for CAM:", chosen_layer)





import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Step 1: Forward pass and capture activations from chosen layer
activations = None
gradients = None

# Hook to capture activations
def forward_hook(module, input, output):
    global activations
    activations = output

# Hook to capture gradients using register_full_backward_hook
def full_backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Register hooks on the chosen layer
chosen_layer.register_forward_hook(forward_hook)
chosen_layer.register_full_backward_hook(full_backward_hook)  # Use full backward hook

# Select one correctly classified sample (e.g., Cargo)
sample_input, sample_label = correct_samples_per_class['Cargo']
sample_input = sample_input.unsqueeze(0)  # Add batch dimension

# Step 2: Forward pass through model
best_model.eval()  # Set model to evaluation mode
output = best_model(sample_input)

# Get predicted class or target class (e.g., class 0 for Cargo)
pred_class = output[1].argmax(dim=1).item()

# Step 3: Backward pass to compute gradients with respect to target class
best_model.zero_grad()  # Zero out previous gradients
target = output[1][0][pred_class]  # Get score for target class (Cargo)
target.backward()  # Backpropagate to compute gradients

# Step 4: Compute weights by averaging gradients across spatial dimensions
weights = torch.mean(gradients, dim=[2, 3])  # Average over H and W

# Step 5: Generate CAM by weighting activations with gradients
cam = torch.sum(weights[:, :, None, None] * activations, dim=1).squeeze()
cam = F.relu(cam)  # Apply ReLU to remove negative values

# Normalize CAM for visualization purposes
cam -= cam.min()
cam /= cam.max()

import torch.nn.functional as F
import matplotlib.pyplot as plt

# Detach CAM from computation graph before converting it to NumPy for visualization
cam = cam.detach().cpu()

# Resize CAM to match logmel_output shape (501 time frames, 64 mel bins)
# We use bilinear interpolation for resizing
cam_resized = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(501, 64), mode='bilinear', align_corners=False)
cam_resized = cam_resized.squeeze().numpy()  # Remove extra dimensions and convert to NumPy

# Visualize resized CAM overlayed on log-mel spectrogram
import matplotlib.pyplot as plt

plt.imshow(logmel_output.squeeze().cpu(), cmap='gray', aspect='auto')  # Plot original log-mel spectrogram
plt.imshow(cam_resized, cmap='jet', alpha=0.5, aspect='auto')  # Overlay resized CAM with transparency (alpha)
plt.colorbar()
# Add labels for the axes
plt.xlabel("Frequency")
plt.ylabel("Time")
# Save the figure with high resolution (e.g., 300 dpi)
plt.savefig('cam_high_res.png', dpi=300)

# Remove plt.show() to avoid displaying the plot
plt.close()  # Close the figure to free up memory


# Hook to capture activations
def forward_hook(module, input, output):
    global activations
    print(f"Activations shape from {module}: {output.shape}")
    activations = output

# Register hook on chosen layer
chosen_layer = best_model.model_ft.backbone.conv_block5.conv2
chosen_layer.register_forward_hook(forward_hook)

# Forward pass through model
sample_input = correct_samples_per_class['Cargo'][0].unsqueeze(0)  # Add batch dimension
best_model.eval()
output = best_model(sample_input)

# Check final output shape
print(f"Final output shape (logits): {output[1].shape}")




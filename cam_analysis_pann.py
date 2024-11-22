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

# Print selected samples per class
for class_name, (input_sample, label) in correct_samples_per_class.items():
    print(f"Selected Sample for {class_name}:")
    print(f"Input Tensor Shape: {input_sample.shape}")
    print(f"Label: {label.item()}")

sample_input, sample_label = correct_samples_per_class['Cargo']
sample_input = sample_input.unsqueeze(0)  # Add batch dimension
with torch.no_grad():
    # Pass through spectrogram extractor and logmel extractor
    spectrogram_output = best_model.model_ft.backbone.spectrogram_extractor(sample_input)
    logmel_output = best_model.model_ft.backbone.logmel_extractor(spectrogram_output)

# Print shapes for debugging
print("Spectrogram Output Shape:", spectrogram_output.shape)  #  [1, 1, 501, 513]
print("Log-Mel Spectrogram Output Shape:", logmel_output.shape)  # [1, 1, 501, 64]

# print('\n')
# # Dictionary to store intermediate outputs
# layer_outputs = {}
# # Hook function to capture input and output of a layer
# def hook_fn(module, input, output):
#     layer_name = module.__class__.__name__  # Get the layer's class name
#     print(f"Layer: {layer_name}")
#     print(f"Input Shape: {tuple(input[0].shape)}")
#     print(f"Output Shape: {tuple(output.shape)}")
#     print("-" * 50)
#     layer_outputs[layer_name] = output  # Store the output for further inspection
# # Attach hooks to all layers in the model
# for name, module in best_model.named_modules():
#     if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
#         module.register_forward_hook(hook_fn)
# # Pass a sample input through the model to trigger hooks
# best_model.eval()
# with torch.no_grad():
#     for batch in test_loader:
#         inputs, labels = batch
#         inputs = inputs.to(device)
#         outputs = best_model(inputs)  # Forward pass (hooks will trigger here)
#         break  # Only process one batch for debugging

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
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    return cam.cpu().detach().numpy()

# Dictionary to store all correctly classified samples per class
correct_samples_per_class = {class_name: [] for class_name in class_to_idx.keys()}
# Populate dictionary with all correctly classified samples
for batch in test_loader:
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = best_model(inputs)
    _, preds = torch.max(outputs[1], dim=1)  # Get predictions

    for i in range(len(labels)):
        if preds[i] == labels[i]:  # Correctly classified
            class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(labels[i].item())]
            correct_samples_per_class[class_name].append((inputs[i], labels[i]))

last_conv_layer = best_model.model_ft.backbone.conv_block6
# Process Grad-CAM for each class
save_single_example = True  # Set this to True to save a single example per class
for class_name, samples in correct_samples_per_class.items():
    print(f"Processing Grad-CAM for {len(samples)} samples in class: {class_name}")

    aggregated_cam = None
    single_example_saved = False  # Track if a single example has been saved

    for idx, (sample_input, sample_label) in enumerate(samples):
        sample_input = sample_input.unsqueeze(0).to(device)  # Add batch dimension
        target_class = sample_label.item()  # Get target class index

        # Compute log-mel spectrogram for current sample
        with torch.no_grad():
            spectrogram_output = best_model.model_ft.backbone.spectrogram_extractor(sample_input)
            logmel_output = best_model.model_ft.backbone.logmel_extractor(spectrogram_output)

        # Generate Grad-CAM heatmap
        cam = generate_gradcam(best_model, sample_input, target_class, last_conv_layer)

        # Resize CAM to match input dimensions (e.g., spectrogram size)
        cam_resized = F.interpolate(torch.tensor(cam).unsqueeze(0).unsqueeze(0), size=(501, 64), mode='bilinear', align_corners=False)
        cam_resized_np = cam_resized.squeeze().numpy()

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

            plt.savefig(f"cam/figures_pann/gradcam_{class_name}_single.png", dpi=300)
            plt.close()

            single_example_saved = True  # Mark that the single example has been saved

    # Average CAM across all samples
    aggregated_cam /= len(samples)

    # Save aggregated CAM visualization
    plt.figure(figsize=(10, 5))
    plt.title(f"Aggregated Grad-CAM Heatmap ({class_name})")
    plt.imshow(logmel_output.squeeze(0).squeeze(0).cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.imshow(aggregated_cam, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.savefig(f"cam/figures_pann/gradcam_{class_name}_aggregated.png", dpi=300)
    plt.close()
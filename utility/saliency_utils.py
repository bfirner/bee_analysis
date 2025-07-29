import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_layer_by_name(model, layer_name):
    """
    Retrieves a layer from the model based on a dot-separated path.
    """
    if isinstance(layer_name, list):
        layer_name = layer_name[0]
    parts = layer_name.split(".")
    layer = model
    for part in parts:
        layer = getattr(layer, part)
    return layer

# Configure logging to use saliency.log
def setup_saliency_logging():
    """Ensure saliency operations log to saliency.log"""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.FileHandler('saliency.log', mode='a')
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def plot_saliency_map(
    model,
    save_folder,
    input_tensor,
    target_class=None,
    batch_num=None,
    model_name="model",
    process_all_samples=True,
    sample_idx=0,
):
    """
    Generates saliency maps for multi-channel (5-frame) input tensor,
    creating separate saliency maps for each frame/channel.
    
    Args:
        process_all_samples: If True, process all samples in batch. If False, process only sample_idx.
        sample_idx: Which sample to process when process_all_samples=False (default: 0)
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    batch_size = input_tensor.shape[0]
    
    if not process_all_samples:
        # Process only one sample
        if sample_idx >= batch_size:
            sample_idx = 0  # Fallback to first sample
        input_tensor = input_tensor[sample_idx:sample_idx+1]
        batch_size = 1
        start_idx = sample_idx
    else:
        start_idx = 0
    
    # Prepare directory
    directory = f"saliency_maps/{save_folder}/"
    os.makedirs(directory, exist_ok=True)
    
    # Process each sample in the (possibly reduced) batch
    for sample_idx_in_batch in range(batch_size):
        # Get single sample for processing
        single_sample = input_tensor[sample_idx_in_batch:sample_idx_in_batch+1]
        single_sample.requires_grad_()
        
        # Forward pass to get predictions
        outputs = model(single_sample)
        pred_class = outputs.argmax(dim=1).item()

        # Determine the target class for saliency
        if target_class is None:
            current_target_class = pred_class
        elif isinstance(target_class, (list, tuple)):
            current_target_class = target_class[start_idx + sample_idx_in_batch]
        else:
            current_target_class = target_class

        # Zero gradients and backward for target class
        model.zero_grad()
        target = outputs[0, current_target_class]
        target.backward()

        # Extract saliency - keep all channels/frames separate
        saliency = single_sample.grad.data.abs().squeeze(0).cpu().numpy()  # Remove batch dimension
        
        # Handle different input shapes
        if saliency.ndim == 2:  # Single channel case (H, W)
            saliency = saliency[np.newaxis, ...]  # Add channel dimension -> (1, H, W)
        elif saliency.ndim == 3:  # Multi-channel case (C, H, W)
            pass  # Already correct shape
        else:
            raise ValueError(f"Unexpected saliency shape: {saliency.shape}")

        # Get number of channels/frames
        num_channels = saliency.shape[0]
        
        # Create subplot for each frame/channel
        fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4))
        
        # Handle single channel case
        if num_channels == 1:
            axes = [axes]

        for i in range(num_channels):
            im = axes[i].imshow(saliency[i], cmap="hot")
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        # Overall title
        actual_sample_idx = start_idx + sample_idx_in_batch
        fig.suptitle(
            f"Saliency Maps - {model_name} Sample {actual_sample_idx} (True: {current_target_class}, Pred: {pred_class})",
            fontsize=14
        )
        
        # Save combined plot
        filename = os.path.join(
            directory,
            f"saliency_map_{model_name}_batch{batch_num}_sample{actual_sample_idx}_true{current_target_class}_pred{pred_class}.png",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        # Also save individual frame saliency maps
        for i in range(num_channels):
            plt.figure(figsize=(6, 6))
            plt.imshow(saliency[i], cmap="hot")
            plt.title(f"Frame {i+1} Saliency - {model_name} Sample {actual_sample_idx} (True: {current_target_class}, Pred: {pred_class})")
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)
            
            frame_filename = os.path.join(
                directory,
                f"saliency_frame{i+1}_{model_name}_batch{batch_num}_sample{actual_sample_idx}_true{current_target_class}_pred{pred_class}.png",
            )
            plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
        # Clear gradients to avoid memory issues
        single_sample.grad = None


def plot_gradcam_for_multichannel_input(
    model,
    save_folder,
    dataset,
    input_tensor,
    target_layer_name,
    model_name,
    target_classes=None,
    number_of_classes=3,
):
    """
    Generates and saves Grad-CAM overlays for each channel in a multi-channel input,
    including true and predicted class annotations.
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Compute model outputs and predicted classes
    with torch.no_grad():
        outputs = model(input_tensor)
    pred_classes = outputs.argmax(dim=1).tolist()

    # Retrieve target layer
    target_layer = get_layer_by_name(model, target_layer_name)

    # If no expected classes provided, use predictions
    if target_classes is None:
        with torch.no_grad():
            target_classes = pred_classes

    # Prepare GradCAM
    targets = [ClassifierOutputTarget(c) for c in target_classes]
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Convert input tensor to numpy for visualization
    input_images = input_tensor.detach().cpu().numpy()

    class_count = {}
    batch_num = 0
    for batch_idx in range(input_images.shape[0]):
        true_class = target_classes[batch_idx]
        pred_class = pred_classes[batch_idx]

        # Track count per true class
        class_count.setdefault(true_class, 0)
        class_count[true_class] += 1
        if class_count[true_class] > 100:
            continue
        if len(class_count) == number_of_classes and all(
                count >= 100 for count in class_count.values()):
            return

        class_directory = f"gradcam_plots/{save_folder}/class_{true_class}/"
        os.makedirs(class_directory, exist_ok=True)

        # Iterate channels
        for channel_idx in range(input_images.shape[1]):
            channel_image = input_images[batch_idx, channel_idx]
            channel_image = (channel_image - channel_image.min()) / (
                channel_image.max() - channel_image.min())
            channel_image_rgb = np.stack([channel_image] * 3, axis=-1)

            cam_image = show_cam_on_image(channel_image_rgb,
                                          grayscale_cam[batch_idx],
                                          use_rgb=True)

            # Plot side by side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(channel_image, cmap="gray")
            axs[0].set_title(
                f"Orig Ch {channel_idx+1}\nTrue: {true_class}, Pred: {pred_class}"
            )
            axs[0].axis("off")

            axs[1].imshow(cam_image)
            axs[1].set_title(
                f"Grad-CAM Ch {channel_idx+1}\nTrue: {true_class}, Pred: {pred_class}"
            )
            axs[1].axis("off")

            filename = os.path.join(
                class_directory,
                f"gradcam_true{true_class}_pred{pred_class}_batch{batch_num}_img{batch_idx}_ch{channel_idx}_layer{target_layer_name}.png",
            )
            plt.savefig(filename)
            plt.close(fig)

        # Also generate a saliency map for the first sample of the batch
        try:
            plot_saliency_map(
                model=model,
                input_tensor=input_tensor,
                target_class=true_class,
                batch_num=batch_num,
                model_name=model_name,
                save_folder=save_folder,
            )
        except Exception as e:
            print(f"Failed to plot saliency map: {e}")

        batch_num += 1

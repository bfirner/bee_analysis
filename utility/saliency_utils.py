import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_layer_by_name(model, layer_name):
    """
    Retrieves a layer from the model based on a dot-separated path.
    """
    if isinstance(layer_name, list):
        layer_name = layer_name[0]
    parts = layer_name.split('.')
    layer = model
    for part in parts:
        layer = getattr(layer, part)
    return layer


def plot_saliency_map(
    model,
    save_folder,
    input_tensor,
    target_class=None,
    batch_num=None,
    model_name="model",
):
    """
    Generates a saliency map for the given input tensor and model,
    annotating both the expected (target) and predicted classes.
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_()

    # Forward pass to get predictions
    outputs = model(input_tensor)
    pred_class = outputs.argmax(dim=1).item()

    # Determine the target class for saliency (expected)
    if target_class is None:
        target_class = pred_class

    # Zero gradients and backward for target class
    model.zero_grad()
    target = outputs[0, target_class]
    target.backward()

    # Extract and process saliency
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    if saliency.ndim == 1:
        saliency = saliency.reshape(
            (input_tensor.shape[2], input_tensor.shape[3]))
    if saliency.ndim > 2:
        saliency = saliency.mean(axis=tuple(range(saliency.ndim - 2)))

    # Prepare directory
    directory = f"saliency_maps/{save_folder}/"
    os.makedirs(directory, exist_ok=True)

    # Plot and save
    plt.figure(figsize=(10, 10))
    plt.imshow(saliency, cmap="hot")
    plt.title(
        f"Saliency Map - {model_name} (True: {target_class}, Pred: {pred_class})")
    plt.axis("off")
    filename = os.path.join(
        directory,
        f"saliency_map_{model_name}_batch{batch_num}_true{target_class}_pred{pred_class}.png"
    )
    plt.savefig(filename)
    plt.close()


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
        if len(class_count) == number_of_classes and all(count >= 100 for count in class_count.values()):
            return

        class_directory = f"gradcam_plots/{save_folder}/class_{true_class}/"
        os.makedirs(class_directory, exist_ok=True)

        # Iterate channels
        for channel_idx in range(input_images.shape[1]):
            channel_image = input_images[batch_idx, channel_idx]
            channel_image = (channel_image - channel_image.min()) / \
                (channel_image.max() - channel_image.min())
            channel_image_rgb = np.stack([channel_image] * 3, axis=-1)

            cam_image = show_cam_on_image(
                channel_image_rgb, grayscale_cam[batch_idx], use_rgb=True)

            # Plot side by side
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(channel_image, cmap="gray")
            axs[0].set_title(
                f"Orig Ch {channel_idx+1}\nTrue: {true_class}, Pred: {pred_class}")
            axs[0].axis("off")

            axs[1].imshow(cam_image)
            axs[1].set_title(
                f"Grad-CAM Ch {channel_idx+1}\nTrue: {true_class}, Pred: {pred_class}")
            axs[1].axis("off")

            filename = os.path.join(
                class_directory,
                f"gradcam_true{true_class}_pred{pred_class}_batch{batch_num}_img{batch_idx}_ch{channel_idx}_layer{target_layer_name}.png"
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

import logging
import os

import torch
import webdataset as wds

# Import or define your models and GradCAM utility:
#
# from models.alexnet import AlexLikeNet
# from models.bennet import BenNet
# from models.resnet import ResNet18, ResNet34
# from models.resnext import ResNext18, ResNext34, ResNext50
# from models.convnext import (ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase)
# from utility.saliency_utils import plot_gradcam_for_multichannel_input
#
# Make sure these imports match your directory structure.


def restore_model(checkpoint_path, net):
    """
    Restores model weights from a given checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path,
                            map_location="cpu",
                            weights_only=False)
    net.load_state_dict(checkpoint["model_dict"])
    logging.info(f"Model weights restored from {checkpoint_path}")


def run_gradcam(
        checkpoint: str,
        dataset_path: str,
        modeltype: str,
        gradcam_cnn_model_layer: list,
        num_images=2,
        sample_frames=1,
        label_offset=1,
        num_outputs=3,
        height=720,
        width=960,
        output_folder=None,
        map_percent=100.0, 
        power_scale=0.8
):
    """
    Runs GradCAM on a given model + dataset using minimal logic.

    Args:
        checkpoint (str): Path to the saved model checkpoint file.
        dataset_path (str): Path to the WebDataset tar file.
        modeltype (str): One of the architectures ["alexnet", "bennet",
            "resnet18", "resnet34", "resnext50", "resnext34", "resnext18",
            "convnextxt", "convnextt", "convnexts", "convnextb"].
        gradcam_cnn_model_layer (list of str): List of layer names to apply GradCAM to.
        num_images (int): Number of samples to load from dataset for GradCAM.
        sample_frames (int): If data is multi-frame, how many frames per sample.
        label_offset (int): If labels in your dataset start at 1 (instead of 0), set offset accordingly.
        num_outputs (int): Number of output classes for the model.
        image_size (tuple): (channels, height, width) shape of each frame.

    Returns:
        None. (GradCAM images are produced by plot_gradcam_for_multichannel_input().)
    """
    # --------------------------------------------------------------------------
    # 1. Prepare device
    # --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # 2. Construct the model
    # --------------------------------------------------------------------------
    # Adjust these imports/definitions to match wherever your model definitions live.
    h, w = height, width

    if modeltype == "alexnet":
        from models.alexnet import AlexLikeNet

        net = AlexLikeNet(
            in_dimensions=(sample_frames, h, w),
            out_classes=num_outputs,
            linear_size=512,
        )
    elif modeltype == "bennet":
        from models.bennet import BenNet

        net = BenNet(in_dimensions=(sample_frames, h, w),
                     out_classes=num_outputs)
    elif modeltype == "resnet18":
        from models.resnet import ResNet18

        net = ResNet18(
            in_dimensions=(sample_frames, h, w),
            out_classes=num_outputs,
            expanded_linear=True,
        )
    elif modeltype == "resnet34":
        from models.resnet import ResNet34

        net = ResNet34(
            in_dimensions=(sample_frames, h, w),
            out_classes=num_outputs,
            expanded_linear=True,
        )
    elif modeltype == "resnext50":
        from models.resnext import ResNext50

        net = ResNext50(
            in_dimensions=(sample_frames, h, w),
            out_classes=num_outputs,
            expanded_linear=True,
        )
    elif modeltype == "resnext34":
        from models.resnext import ResNext34

        net = ResNext34(
            in_dimensions=(sample_frames, h, w),
            out_classes=num_outputs,
            expanded_linear=False,
            use_dropout=False,
        )
    elif modeltype == "resnext18":
        from models.resnext import ResNext18

        net = ResNext18(
            in_dimensions=(sample_frames, h, w),
            out_classes=num_outputs,
            expanded_linear=True,
            use_dropout=False,
        )
    elif modeltype == "convnextxt":
        from models.convnext import ConvNextExtraTiny

        net = ConvNextExtraTiny(in_dimensions=(sample_frames, h, w),
                                out_classes=num_outputs)
    elif modeltype == "convnextt":
        from models.convnext import ConvNextTiny

        net = ConvNextTiny(in_dimensions=(sample_frames, h, w),
                           out_classes=num_outputs)
    elif modeltype == "convnexts":
        from models.convnext import ConvNextSmall

        net = ConvNextSmall(in_dimensions=(sample_frames, h, w),
                            out_classes=num_outputs)
    elif modeltype == "convnextb":
        from models.convnext import ConvNextBase

        net = ConvNextBase(in_dimensions=(sample_frames, h, w),
                           out_classes=num_outputs)
    else:
        raise ValueError(f"Unknown model type: {modeltype}")

    net.to(device)
    net.eval()

    # --------------------------------------------------------------------------
    # 3. Restore weights
    # --------------------------------------------------------------------------
    restore_model(checkpoint, net)

    # --------------------------------------------------------------------------
    # 4. Build a small DataLoader for the WebDataset
    # --------------------------------------------------------------------------
    # Minimal decode: we assume each sample has N frames (like 0.png, 1.png, etc.)
    # plus a 'cls' label. Adjust the keys if your data is different.
    decode_strs = [f"{i}.png" for i in range(sample_frames)] + ["cls"]

    dataset = (
        wds.
        WebDataset(dataset_path, shardshuffle=20000 // sample_frames).decode(
            "l")  # decode as grayscale images; adjust if you have color data
        .to_tuple(*decode_strs))

    # We'll just load one small batch with `num_images` items:
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=num_images,
                                         num_workers=64)

    # --------------------------------------------------------------------------
    # 5. Forward pass and GradCAM
    # --------------------------------------------------------------------------
    # We'll do only one iteration (i.e., one batch).
    from utility.saliency_utils import plot_gradcam_for_multichannel_input

    # Use the output_folder if provided; otherwise, use the dataset's basename.
    save_folder = (os.path.join(output_folder, os.path.basename(dataset_path)) if output_folder is not None else
                   os.path.basename(dataset_path))
    
    for batch in loader:
        # Unpack the batch: frames + label
        *frames, labels = batch
        
        # Stack frames into a tensor: (batch_size, num_frames, height, width)
        net_input = torch.stack(frames, dim=1).float().to(device)
        
        # Process labels: convert to tensor and apply offset
        labels = torch.tensor(labels).long().to(device) - label_offset
        
        logging.info(f"Processing batch with input shape: {net_input.shape}, labels: {labels}")
        # END OF ADDITION
        
        # For demonstration, you might run GradCAM on each layer in gradcam_cnn_model_layer.
        with torch.set_grad_enabled(True):
            for layer_name in gradcam_cnn_model_layer:
                try:
                    logging.info(
                        f"Running GradCAM for layer {layer_name} in folder {save_folder}..."
                    )
                    plot_gradcam_for_multichannel_input(
                        model=net,
                        save_folder=save_folder,
                        dataset=dataset,
                        input_tensor=net_input,
                        target_layer_name=layer_name, 
                        model_name=modeltype,
                        target_classes=labels.tolist(),
                        number_of_classes=num_outputs,
                        map_percent=map_percent,
                    )
                except Exception as e:
                    logging.info(
                        f"Error plotting GradCAM for layer {layer_name}: {e}")
        break  # Process only the first batch and stop.

    logging.info("GradCAM process completed.")
#!/usr/bin/env python3
"""
Manual GradCAM and Saliency Map Generator

This script generates GradCAM and saliency maps from a pre-trained model checkpoint
without requiring retraining. It loads the model, processes datasets, and generates
visualization outputs for multiple class tar files.

Usage:
    python generate_visualizations.py --checkpoint model.checkpoint --dataset_dir ./dataset_tar_files --output_dir ./visualizations
    python generate_visualizations.py --checkpoint model.checkpoint --datasets class0.tar class1.tar class2.tar --output_dir ./visualizations
"""

import argparse
import logging
import os
import sys
import torch
import webdataset as wds
from pathlib import Path
import glob
import random

# Add the bee_analysis directory to the path for imports
script_dir = Path(__file__).parent.absolute()
# Since we're already in bee_analysis, just add current directory
sys.path.insert(0, str(script_dir))

try:
    from models.alexnet import AlexLikeNet
    from models.bennet import BenNet
    from models.resnet import ResNet18, ResNet34
    from models.resnext import ResNext18, ResNext34, ResNext50
    from models.convnext import ConvNextExtraTiny, ConvNextTiny, ConvNextSmall, ConvNextBase
    from utility.saliency_utils import plot_gradcam_for_multichannel_input, plot_saliency_map
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Make sure you're running this script from the correct directory")
    sys.exit(1)


def setup_logging(debug=False, log_file=None):
    """Setup logging configuration with file output"""
    level = logging.DEBUG if debug else logging.INFO
    
    # Set log file path - 2 levels above bee_analysis directory
    if log_file is None:
        script_dir = Path(__file__).parent.absolute()  # bee_analysis
        parent_dir = script_dir.parent.parent  # 2 levels up
        log_file = parent_dir / "visualization.log"
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Create console handler for immediate feedback (less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Always INFO for console
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[file_handler, console_handler]
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('pytorch_grad_cam').setLevel(logging.WARNING)
    logging.getLogger('torchvision').setLevel(logging.WARNING)
    
    logging.info(f"Logging to file: {log_file}")


def find_dataset_files(dataset_dir=None, dataset_files=None):
    """Find all dataset tar files"""
    if dataset_files:
        # Explicit list of files provided
        tar_files = []
        for file_path in dataset_files:
            if os.path.exists(file_path):
                tar_files.append(file_path)
                logging.info(f"Found dataset file: {file_path}")
            else:
                logging.warning(f"Dataset file not found: {file_path}")
        return tar_files
    
    elif dataset_dir:
        # Search for tar files in directory
        if not os.path.exists(dataset_dir):
            logging.error(f"Dataset directory not found: {dataset_dir}")
            return []
        
        tar_files = glob.glob(os.path.join(dataset_dir, "*.tar"))
        tar_files.sort()  # Sort for consistent ordering
        
        logging.info(f"Found {len(tar_files)} tar files in {dataset_dir}:")
        for tar_file in tar_files:
            logging.info(f"  - {tar_file}")
        
        return tar_files
    
    else:
        logging.error("Either dataset_dir or dataset_files must be provided")
        return []


def load_checkpoint(checkpoint_path):
    """Load and return checkpoint data"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise


def create_model(modeltype, model_args):
    """Create and return the appropriate model based on modeltype"""
    try:
        if modeltype == "alexnet":
            net = AlexLikeNet(**model_args)
        elif modeltype == "bennet":
            net = BenNet(**model_args)
        elif modeltype == "resnet18":
            net = ResNet18(**model_args)
        elif modeltype == "resnet34":
            net = ResNet34(**model_args)
        elif modeltype == "resnext50":
            net = ResNext50(**model_args)
        elif modeltype == "resnext34":
            net = ResNext34(**model_args)
        elif modeltype == "resnext18":
            net = ResNext18(**model_args)
        elif modeltype == "convnextxt":
            net = ConvNextExtraTiny(**model_args)
        elif modeltype == "convnextt":
            net = ConvNextTiny(**model_args)
        elif modeltype == "convnexts":
            net = ConvNextSmall(**model_args)
        elif modeltype == "convnextb":
            net = ConvNextBase(**model_args)
        else:
            raise ValueError(f"Unknown model type: {modeltype}")
        
        logging.info(f"Created model of type: {modeltype}")
        return net
    except Exception as e:
        logging.error(f"Failed to create model of type {modeltype}: {e}")
        raise


def create_dataset(dataset_path, sample_frames, batch_size=32, num_workers=4):
    """Create and return dataset and dataloader"""
    try:
        # Build decode strings for frames and labels
        decode_strs = [f"{i}.png" for i in range(sample_frames)] + ["cls"]
        
        logging.info(f"Creating dataset from {dataset_path}")
        logging.debug(f"Decode strings: {decode_strs}")
        
        dataset = (
            wds.WebDataset(dataset_path, shardshuffle=20000 // sample_frames, empty_check=False)  # Add empty_check=False
            .decode("l")  # decode as grayscale images
            .to_tuple(*decode_strs)
        )
        
        # Reduce num_workers for small datasets
        effective_workers = min(num_workers, 1) if "dataset_0" in dataset_path else num_workers
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=effective_workers,  # Use reduced workers
            drop_last=False
        )
        
        logging.info(f"Created dataset with batch_size={batch_size}, num_workers={effective_workers}")
        return dataset, dataloader
        
    except Exception as e:
        logging.error(f"Failed to create dataset from {dataset_path}: {e}")
        raise


def get_gradcam_layers(modeltype):
    """Return default GradCAM layers for each model type"""
    layer_mapping = {
        "alexnet": ["model_a.4.0", "model_b.4.0"],
        "bennet": ["model_a.4.0", "model_b.4.0"],
        "resnet18": ["layer4.1.conv2"],
        "resnet34": ["layer4.2.conv2"],
        "resnext18": ["layer4.1.conv2"],
        "resnext34": ["layer4.2.conv2"],
        "resnext50": ["layer4.2.conv3"],
        "convnextxt": ["stages.3.blocks.1.pwconv2"],
        "convnextt": ["stages.3.blocks.2.pwconv2"],
        "convnexts": ["stages.3.blocks.26.pwconv2"],
        "convnextb": ["stages.3.blocks.26.pwconv2"],
    }
    return layer_mapping.get(modeltype, ["layer4"])




def process_batch(model, input_tensor, labels, batch_idx, args, metadata, dataset_name):
    """Process a single batch for GradCAM and saliency map generation"""
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    labels = labels.to(device)
    
    batch_sample_id = hash(f"{dataset_name}_{batch_idx}")
    if not should_process_sample(args.map_percent, batch_sample_id):
        logging.debug(f"Skipping batch {batch_idx} from {dataset_name} due to map_percent={args.map_percent}")
        return
    
    logging.info(f"Processing batch {batch_idx} from {dataset_name} (selected by map_percent={args.map_percent})")
    # END OF ADDITION
    
    # Adjust labels by offset
    adjusted_labels = labels - args.label_offset
    
    # Create save folder directly in the saliency_maps directory
    save_folder = os.path.join("saliency_maps", dataset_name)
    
    # Generate GradCAM for each specified layer
    if args.generate_gradcam:
        for layer_name in args.gradcam_layers:
            try:
                logging.info(f"Generating GradCAM for layer {layer_name}, dataset {dataset_name}, batch {batch_idx}")
                plot_gradcam_for_multichannel_input(
                    model=model,
                    save_folder=save_folder,
                    dataset=None,  # Not used in the function
                    input_tensor=input_tensor,
                    target_layer_name=layer_name,
                    model_name=metadata['modeltype'],
                    target_classes=adjusted_labels.tolist(),
                    number_of_classes=metadata['label_size'],
                    map_percent=args.map_percent,
                )
                logging.info(f"Successfully generated GradCAM for layer {layer_name}")
            except Exception as e:
                logging.error(f"Failed to generate GradCAM for layer {layer_name}: {e}")
                if args.debug:
                    logging.exception("Full traceback:")
    
    # Generate Saliency Maps - existing code unchanged
    if args.generate_saliency:
        try:
            logging.info(f"Generating saliency maps for dataset {dataset_name}, batch {batch_idx}")
            plot_saliency_map(
                model=model,
                save_folder=save_folder,
                input_tensor=input_tensor,
                target_class=adjusted_labels.tolist(),
                batch_num=batch_idx,
                model_name=metadata['modeltype'],
                process_all_samples=args.process_all_samples,
                sample_idx=args.sample_idx,
                map_percent=args.map_percent,
            )
            logging.info(f"Successfully generated saliency maps for dataset {dataset_name}")
        except Exception as e:
            logging.error(f"Failed to generate saliency maps for dataset {dataset_name}: {e}")
            if args.debug:
                logging.exception("Full traceback:")

def process_dataset(model, dataset_path, args, metadata):
    """Process a single dataset tar file"""
    dataset_name = os.path.basename(dataset_path).replace('.tar', '')
    logging.info(f"Processing dataset: {dataset_name}")
    
    try:
        # Create dataset
        dataset, dataloader = create_dataset(
            dataset_path, 
            metadata['model_args']['in_dimensions'][0], 
            args.batch_size, 
            args.num_workers
        )
        
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                logging.info(f"Processing batch {batch_idx + 1} from {dataset_name}")
                
                # Extract input tensor and labels
                sample_frames = metadata['model_args']['in_dimensions'][0]
                if sample_frames == 1:
                    input_tensor = batch[0].unsqueeze(1)  # Add channel dimension
                    labels = batch[1]
                else:
                    # Concatenate multiple frames
                    frames = []
                    for i in range(sample_frames):
                        frames.append(batch[i].unsqueeze(1))
                    input_tensor = torch.cat(frames, dim=1)
                    labels = batch[sample_frames]
                
                logging.debug(f"Input tensor shape: {input_tensor.shape}")
                logging.debug(f"Labels shape: {labels.shape}")
                
                # Process the batch
                process_batch(model, input_tensor, labels, batch_idx, args, metadata, dataset_name)
                
                batch_count += 1
                
                # Check if we've processed enough batches for this dataset
                if args.num_batches > 0 and batch_count >= args.num_batches:
                    logging.info(f"Processed {batch_count} batches from {dataset_name} as requested")
                    break
                    
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx} from {dataset_name}: {e}")
                if args.debug:
                    logging.exception("Full traceback:")
                continue
        
        logging.info(f"Completed processing {batch_count} batches from {dataset_name}")
        return batch_count
        
    except Exception as e:
        logging.error(f"Error processing dataset {dataset_name}: {e}")
        if args.debug:
            logging.exception("Full traceback:")
        return 0
    

def should_process_sample(map_percent, sample_id=None):
    """
    Randomly determine if a sample should be processed based on map_percent.
    
    Args:
        map_percent: Percentage of samples to process (0-100)
        sample_id: Optional unique identifier for deterministic selection
    
    Returns:
        bool: True if sample should be processed
    """
    if map_percent >= 100.0:
        return True
    if map_percent <= 0.0:
        return False
    
    # Use sample_id for deterministic selection if provided
    if sample_id is not None:
        # Use modulo approach for more predictable results with low percentages
        return (sample_id % 100) < map_percent
    
    return random.random() * 100.0 < map_percent

def main():
    parser = argparse.ArgumentParser(
        description="Generate GradCAM and saliency maps from a trained model checkpoint"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file"
    )
    
    # Mutually exclusive group for dataset specification
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory containing dataset tar files (will process all .tar files)"
    )
    
    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        help="List of specific dataset tar files to process"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualization outputs (default: ./visualizations)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    
    parser.add_argument(
        "--num_batches",
        type=int,
        default=5,
        help="Number of batches to process per dataset (default: 5, -1 for all)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)"
    )
    
    parser.add_argument(
        "--gradcam_layers",
        nargs="+",
        default=None,
        help="Specific layers for GradCAM (if not provided, uses model defaults)"
    )
    
    parser.add_argument(
        "--label_offset",
        type=int,
        default=0,
        help="Label offset for adjustment (default: 0)"
    )
    
    parser.add_argument(
        "--generate_gradcam",
        action="store_true",
        default=True,
        help="Generate GradCAM visualizations (default: True)"
    )
    
    parser.add_argument(
        "--generate_saliency",
        action="store_true",
        default=True,
        help="Generate saliency maps (default: True)"
    )
    
    parser.add_argument(
        "--no_gradcam",
        action="store_true",
        help="Disable GradCAM generation"
    )
    
    parser.add_argument(
        "--no_saliency",
        action="store_true",
        help="Disable saliency map generation"
    )
    
    parser.add_argument(
        "--process_all_samples",
        action="store_true",
        default=True,
        help="Process all samples in batch for saliency maps (default: True)"
    )
    
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index to process when not processing all samples (default: 0)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
    "--map_percent",
    type=float,
    required=False,
    default=12.5,
    help="Percentage of samples to use for saliency maps and GradCAM (0-100, default: 12.5)",
)
    
    args = parser.parse_args()

     # Validate map_percent
    if args.map_percent < 0 or args.map_percent > 100:
        logging.error(f"map_percent must be between 0 and 100, got {args.map_percent}")
        sys.exit(1)

    
    # Handle negation flags
    if args.no_gradcam:
        args.generate_gradcam = False
    if args.no_saliency:
        args.generate_saliency = False
    
    # Setup logging
    setup_logging(args.debug)
    
    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Find dataset files
    if args.dataset_dir:
        tar_files = find_dataset_files(dataset_dir=args.dataset_dir)
    else:
        tar_files = find_dataset_files(dataset_files=args.datasets)
    
    if not tar_files:
        logging.error("No dataset files found")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load checkpoint
        logging.info("Loading checkpoint...")
        checkpoint = load_checkpoint(args.checkpoint)
        metadata = checkpoint['metadata']
        
        # Log model information
        logging.info(f"Model type: {metadata['modeltype']}")
        logging.info(f"Label size: {metadata['label_size']}")
        logging.info(f"Model args: {metadata['model_args']}")
        
        # Create model
        logging.info("Creating model...")
        model = create_model(metadata['modeltype'], metadata['model_args'])
        
        # Load model weights
        model.load_state_dict(checkpoint['model_dict'])
        model.eval()
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logging.info(f"Using device: {device}")
        
        # Determine sample frames from model args
        sample_frames = metadata['model_args']['in_dimensions'][0]
        logging.info(f"Sample frames: {sample_frames}")
        
        # Set GradCAM layers if not provided
        if args.gradcam_layers is None:
            args.gradcam_layers = get_gradcam_layers(metadata['modeltype'])
        logging.info(f"GradCAM layers: {args.gradcam_layers}")
        
        # Process each dataset
        total_batches_processed = 0
        for tar_file in tar_files:
            logging.info(f"\n{'='*60}")
            logging.info(f"Starting processing of {tar_file}")
            logging.info(f"{'='*60}")
            
            batches_processed = process_dataset(model, tar_file, args, metadata)
            total_batches_processed += batches_processed
            
            logging.info(f"Finished processing {tar_file}: {batches_processed} batches")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"SUMMARY")
        logging.info(f"{'='*60}")
        logging.info(f"Processed {len(tar_files)} dataset files")
        logging.info(f"Total batches processed: {total_batches_processed}")
        logging.info(f"Visualizations saved to: {args.output_dir}")
        
        # List output directories
        for tar_file in tar_files:
            dataset_name = os.path.basename(tar_file).replace('.tar', '')
            output_subdir = os.path.join(args.output_dir, dataset_name)
            if os.path.exists(output_subdir):
                file_count = len([f for f in os.listdir(output_subdir) if f.endswith('.png')])
                logging.info(f"  {dataset_name}: {file_count} visualization files")
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if args.debug:
            logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
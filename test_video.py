#!/usr/bin/env python3
"""
Video inference script for JIST (Joint Image and Sequence Training)
This script takes a video file as input and extracts sequential visual descriptors
using a trained JIST model.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from jist import utils
from jist.models import JistModel


def extract_frames_from_video(video_path, sample_rate=1, max_frames=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        sample_rate: Extract every Nth frame (default: 1 = all frames)
        max_frames: Maximum number of frames to extract (None = all)
    
    Returns:
        List of frames as numpy arrays (RGB format)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logging.info(f"Video info: {total_frames} frames at {fps:.2f} FPS")
    logging.info(f"Extracting every {sample_rate} frame(s)")
    
    frames = []
    frame_idx = 0
    frames_extracted = 0
    
    pbar = tqdm(total=total_frames, desc="Extracting frames", ncols=100)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frames_extracted += 1
            
            if max_frames and frames_extracted >= max_frames:
                break
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    logging.info(f"Extracted {len(frames)} frames from video")
    return frames


def preprocess_frames(frames, transform):
    """
    Apply transformations to frames.
    
    Args:
        frames: List of numpy arrays (RGB format)
        transform: Torchvision transform
    
    Returns:
        List of transformed PIL images
    """
    pil_frames = [Image.fromarray(frame) for frame in frames]
    transformed_frames = [transform(frame) for frame in pil_frames]
    return transformed_frames


def create_sequences(frames, seq_length, stride=1):
    """
    Create overlapping sequences from frames.
    
    Args:
        frames: List of preprocessed frame tensors
        seq_length: Number of frames per sequence
        stride: Stride between consecutive sequences
    
    Returns:
        List of sequences (each sequence is a list of frames)
    """
    if len(frames) < seq_length:
        logging.warning(f"Video has only {len(frames)} frames, but seq_length is {seq_length}")
        logging.warning(f"Padding with repeated frames")
        while len(frames) < seq_length:
            frames.append(frames[-1])
    
    sequences = []
    for i in range(0, len(frames) - seq_length + 1, stride):
        seq = frames[i:i + seq_length]
        sequences.append(seq)
    
    return sequences


def inference_on_video(args):
    """
    Main inference function for video processing.
    """
    start_time = datetime.now()
    
    # Setup output folder
    video_name = Path(args.video_path).stem
    args.output_folder = f"test_video/{args.exp_name}/{video_name}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    utils.setup_logging(args.output_folder, console="info")
    
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")
    
    # Load the model
    logging.info("Loading JIST model...")
    try:
        model = JistModel(args, agg_type=args.aggregation_type)
    except Exception as e:
        logging.error(f"Failed to load model architecture from torch hub: {e}")
        logging.error("This might be due to network issues. Retrying...")
        try:
            # Clear torch hub cache and retry
            import time
            time.sleep(2)
            model = JistModel(args, agg_type=args.aggregation_type)
        except Exception as e2:
            logging.error(f"Retry failed: {e2}")
            logging.error("\nTroubleshooting steps:")
            logging.error("1. Check your internet connection")
            logging.error("2. Try manually downloading the cosplace model:")
            logging.error("   python -c \"import torch; torch.hub.load('gmberton/cosplace', 'get_trained_model', backbone='ResNet18', fc_output_dim=512, trust_repo=True)\"")
            logging.error("3. Or use a pre-downloaded model with --pretrained_backbone")
            raise
    
    if args.resume_model is None:
        raise ValueError("Please provide a trained model path using --resume_model")
    
    logging.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model, map_location=args.device)
    model.load_state_dict(model_state_dict)
    model = model.to(args.device)
    model.eval()
    
    logging.info(f"Model loaded. Feature dim: {model.fc_output_dim}, Aggregation dim: {model.aggregation_dim}")
    
    # Setup transforms
    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img_shape = (args.img_shape[0], args.img_shape[1])
    transform = utils.configure_transform(image_dim=img_shape, meta=meta)
    
    # Extract frames from video
    logging.info(f"Processing video: {args.video_path}")
    frames = extract_frames_from_video(
        args.video_path,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames
    )
    
    # Preprocess frames
    logging.info("Preprocessing frames...")
    processed_frames = preprocess_frames(frames, transform)
    
    # Create sequences
    logging.info(f"Creating sequences with length {args.seq_length} and stride {args.stride}...")
    sequences = create_sequences(processed_frames, args.seq_length, args.stride)
    logging.info(f"Created {len(sequences)} sequences")
    
    # Run inference
    logging.info("Running inference...")
    all_descriptors = []
    
    with torch.no_grad():
        for seq_idx, sequence in enumerate(tqdm(sequences, desc="Processing sequences", ncols=100)):
            # Stack frames in sequence: (seq_length, C, H, W)
            seq_tensor = torch.stack(sequence).to(args.device)
            
            # Forward through model
            frames_features = model(seq_tensor)
            aggregated_features = model.aggregate(frames_features)
            
            # Store descriptor
            descriptor = aggregated_features.cpu().numpy()
            all_descriptors.append(descriptor[0])  # Remove batch dimension
    
    # Convert to numpy array
    all_descriptors = np.array(all_descriptors)
    logging.info(f"Generated {all_descriptors.shape[0]} descriptors of dimension {all_descriptors.shape[1]}")
    
    # Save descriptors
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    descriptors_file = output_path / "descriptors.npy"
    np.save(descriptors_file, all_descriptors)
    logging.info(f"Saved descriptors to {descriptors_file}")
    
    # Save metadata
    metadata = {
        'video_path': str(args.video_path),
        'num_frames_extracted': len(frames),
        'num_sequences': len(sequences),
        'seq_length': args.seq_length,
        'stride': args.stride,
        'sample_rate': args.sample_rate,
        'descriptor_dim': all_descriptors.shape[1],
        'aggregation_type': args.aggregation_type,
        'model_path': args.resume_model,
    }
    
    import json
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata to {metadata_file}")
    
    # Optional: compute self-similarity matrix for visualization
    if args.compute_similarity:
        logging.info("Computing self-similarity matrix...")
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(all_descriptors)
        
        similarity_file = output_path / "similarity_matrix.npy"
        np.save(similarity_file, similarity_matrix)
        logging.info(f"Saved similarity matrix to {similarity_file}")
        
        # Save a visualization
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(similarity_matrix, cmap='viridis')
            plt.colorbar(label='Cosine Similarity')
            plt.title(f'Self-Similarity Matrix for {video_name}')
            plt.xlabel('Sequence Index')
            plt.ylabel('Sequence Index')
            
            plot_file = output_path / "similarity_matrix.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved similarity plot to {plot_file}")
        except ImportError:
            logging.warning("matplotlib not available, skipping similarity visualization")
    
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
    logging.info(f"\nOutput summary:")
    logging.info(f"  - Descriptors: {descriptors_file}")
    logging.info(f"  - Metadata: {metadata_file}")
    if args.compute_similarity:
        logging.info(f"  - Similarity matrix: {similarity_file}")


def parse_video_arguments():
    """Parse command line arguments for video inference."""
    parser = argparse.ArgumentParser(
        description="JIST Video Inference - Extract sequential visual descriptors from video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--resume_model", type=str, required=True,
                        help="Path to trained JIST model checkpoint")
    
    # Video processing parameters
    parser.add_argument("--sample_rate", type=int, default=5,
                        help="Extract every Nth frame (default 5 = keep 1 of every 5 frames)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to extract (None = all)")
    
    # Sequence parameters
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Number of frames in each sequence")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride between consecutive sequences")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="ResNet18",
                        choices=["ResNet18", "ResNet50", "ResNet101", "VGG16"],
                        help="Backbone architecture")
    parser.add_argument("--aggregation_type", type=str, default="seqgem",
                        choices=["concat", "mean", "max", "simplefc", "conv1d", "meanfc", "seqgem"],
                        help="Sequence aggregation method")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of frame descriptor")
    parser.add_argument('--img_shape', type=int, default=[480, 640], nargs=2,
                        help="Resizing shape for images (HxW)")
    
    # Output parameters
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Experiment name for output folder")
    parser.add_argument("--compute_similarity", action='store_true',
                        help="Compute and save self-similarity matrix")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    
    # Validate model path
    model_path = Path(args.resume_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.resume_model}")
    
    return args


if __name__ == "__main__":
    args = parse_video_arguments()
    inference_on_video(args)

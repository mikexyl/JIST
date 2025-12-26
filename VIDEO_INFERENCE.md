# JIST Video Inference

This script allows you to run JIST inference on video files to extract sequential visual place recognition descriptors.

## Installation

In addition to the standard JIST requirements, you'll need:

```bash
pip install opencv-python scikit-learn matplotlib
```

## Basic Usage

### Extract descriptors from a video

```bash
python test_video.py \
    --video_path /path/to/your/video.mp4 \
    --resume_model /path/to/trained/model.pth \
    --exp_name my_video_test
```

### Advanced Options

#### Control frame sampling
```bash
# Extract every 5th frame (default; useful for high FPS videos)
python test_video.py \
    --video_path /path/to/video.mp4 \
    --resume_model /path/to/model.pth \
    --sample_rate 5

# Limit to first 1000 frames
python test_video.py \
    --video_path /path/to/video.mp4 \
    --resume_model /path/to/model.pth \
    --max_frames 1000
```

#### Adjust sequence parameters
```bash
# Use sequences of 10 frames with stride of 5
python test_video.py \
    --video_path /path/to/video.mp4 \
    --resume_model /path/to/model.pth \
    --seq_length 10 \
    --stride 5
```

#### Compute self-similarity matrix
```bash
# Generate similarity matrix and visualization
python test_video.py \
    --video_path /path/to/video.mp4 \
    --resume_model /path/to/model.pth \
    --compute_similarity
```

#### Use different model configurations
```bash
# Use ResNet50 backbone with mean aggregation
python test_video.py \
    --video_path /path/to/video.mp4 \
    --resume_model /path/to/model.pth \
    --backbone ResNet50 \
    --aggregation_type mean \
    --fc_output_dim 2048
```

## Output

The script creates an output folder at `test_video/<exp_name>/<video_name>_<timestamp>/` containing:

1. **descriptors.npy**: NumPy array of shape `(num_sequences, descriptor_dim)` containing the sequential descriptors
2. **metadata.json**: Information about the video processing (frame count, sequence parameters, etc.)
3. **similarity_matrix.npy** (optional): Self-similarity matrix for the video
4. **similarity_matrix.png** (optional): Visualization of the similarity matrix
5. **log.txt**: Detailed log of the processing

## Use Cases

### 1. Visual Place Recognition
Extract descriptors and compare against a database of known locations:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load descriptors from video
query_descriptors = np.load('test_video/my_test/video_2024-01-01/descriptors.npy')

# Load database descriptors (from previous processing)
db_descriptors = np.load('database_descriptors.npy')

# Find most similar locations
similarities = cosine_similarity(query_descriptors, db_descriptors)
top_matches = np.argmax(similarities, axis=1)
```

### 2. Loop Closure Detection
Detect when the camera returns to previously visited locations:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load similarity matrix
similarity = np.load('test_video/my_test/video/similarity_matrix.npy')

# Find loop closures (high similarity between distant frames)
threshold = 0.8
for i in range(len(similarity)):
    for j in range(i + 50, len(similarity)):  # Skip nearby frames
        if similarity[i, j] > threshold:
            print(f"Loop closure detected: frame {i} matches frame {j}")
```

### 3. Video Summarization
Identify distinct segments in the video based on visual similarity:

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Load descriptors
descriptors = np.load('test_video/my_test/video/descriptors.npy')

# Cluster similar sequences
clustering = AgglomerativeClustering(n_clusters=10)
labels = clustering.fit_predict(descriptors)

# Each cluster represents a visually distinct segment
print(f"Video contains {len(np.unique(labels))} distinct segments")
```

## Parameters Reference

### Required
- `--video_path`: Path to input video file
- `--resume_model`: Path to trained JIST model checkpoint (.pth file)

### Video Processing
- `--sample_rate`: Extract every Nth frame (default: 1)
- `--max_frames`: Maximum frames to extract (default: None = all)

### Sequence Parameters
- `--seq_length`: Frames per sequence (default: 5)
- `--stride`: Stride between sequences (default: 1)

### Model Configuration
- `--backbone`: Backbone network (default: ResNet18)
  - Options: ResNet18, ResNet50, ResNet101, VGG16
- `--aggregation_type`: Sequence aggregation method (default: seqgem)
  - Options: concat, mean, max, simplefc, conv1d, meanfc, seqgem
- `--fc_output_dim`: Frame descriptor dimension (default: 512)
- `--img_shape`: Image resize shape HxW (default: 480 640)

### Output
- `--exp_name`: Experiment name (default: "default")
- `--compute_similarity`: Compute self-similarity matrix (flag)

### Hardware
- `--device`: cuda or cpu (default: cuda)

## Tips

1. **For high FPS videos**: Use `--sample_rate` to reduce processing time
2. **For long videos**: Use `--max_frames` to process only a portion
3. **For dense matching**: Use `--stride 1` (default)
4. **For faster processing**: Use `--stride` equal to `--seq_length` for non-overlapping sequences
5. **Memory issues**: Reduce `--img_shape` or process in smaller chunks using `--max_frames`

## Examples

### Process a 30 FPS driving video
```bash
python test_video.py \
    --video_path driving_video.mp4 \
    --resume_model models/jist_best.pth \
    --sample_rate 10 \
    --seq_length 5 \
    --stride 3 \
    --compute_similarity
```

### Quick test on short clip
```bash
python test_video.py \
    --video_path test_clip.mp4 \
    --resume_model models/jist_best.pth \
    --max_frames 500 \
    --exp_name quick_test
```

### High-quality processing for matching
```bash
python test_video.py \
    --video_path query_video.mp4 \
    --resume_model models/jist_best.pth \
    --sample_rate 1 \
    --seq_length 10 \
    --stride 1 \
    --backbone ResNet50 \
    --fc_output_dim 2048
```

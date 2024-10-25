# Image Segmentation using Attention-Enhanced Autoencoder and K-means

A sophisticated image segmentation system that combines deep learning (autoencoder with attention mechanism) and traditional clustering (k-means) to perform advanced image segmentation. The system reduces dimensionality while preserving important features through attention mechanisms, followed by clustering to identify distinct segments in the image.

## Features

- **Attention-Enhanced Autoencoder**
  - Multi-head self-attention mechanism
  - Dense encoding/decoding layers
  - Custom architecture for color feature learning

- **K-means Clustering Integration**
  - Clustering on encoded features
  - Automatic segment identification
  - Configurable number of clusters

- **Visualization Tools**
  - 3D color space visualization
  - Cluster distribution plots
  - Side-by-side comparison of original and segmented images

## Requirements

```bash
numpy
matplotlib
scikit-learn
opencv-python
tensorflow
```

## Project Structure

```
├── utils.py           # Utility functions and model architecture
├── main.py           # Main execution script
└── assets/           # Directory for input images
    └── lena_color.png
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install numpy matplotlib scikit-learn opencv-python tensorflow 
```

## Usage

### Basic Usage

```python
from utils import *

# Load and process image
image_path = "path/to/your/image.png"
main(image_path, n_clusters=4)
```

### Advanced Usage

```python
# Custom number of clusters
main(image_path, n_clusters=6)  # For more detailed segmentation

# Access individual components
original_img = load_image(image_path)
img_data = reshape_image(original_img)
autoencoder = build_autoencoder()
```

## Model Architecture

### Autoencoder with Attention

1. **Input Layer**: Accepts RGB values (3 dimensions)
2. **Encoder**:
   - Dense layer (64 units)
   - Multi-Head Attention layer (2 heads)
   - Dense layer (32 units)
3. **Decoder**:
   - Dense layer (64 units)
   - Output layer (3 units)

### Key Components

1. **Image Loading and Preprocessing**
   ```python
   load_image()  # Loads and converts BGR to RGB
   reshape_image()  # Prepares image for processing
   ```

2. **Autoencoder Training**
   ```python
   train_autoencoder()  # Trains for specified epochs
   ```

3. **Clustering and Visualization**
   ```python
   perform_kmeans()  # Clusters encoded data
   plot_3d_scatter()  # Visualizes color distribution
   ```

## Visualization Features

The system provides three types of visualizations:
1. Original color distribution in 3D space
2. Encoded data clustering visualization
3. Side-by-side comparison of original and segmented images

## Performance Optimization

- Batch processing for autoencoder training
- Efficient numpy operations for image manipulation
- Optimized k-means implementation from scikit-learn

## Customization Options

1. **Autoencoder Architecture**
   - Modify number of layers
   - Adjust attention heads
   - Change activation functions

2. **Clustering Parameters**
   - Adjust number of clusters
   - Modify k-means parameters

3. **Training Parameters**
   - Change number of epochs
   - Adjust batch size
   - Modify learning rate

## Example Output

The system generates:
- Original image visualization
- Segmented image result
- 3D scatter plots of color distribution
- Cluster distribution visualization

for n_clusters = 4:
![Screenshot 2024-10-25 190142](https://github.com/user-attachments/assets/ab3f9b2a-cbec-4fce-bac5-ef02d9cd6792)

![Screenshot 2024-10-25 190216](https://github.com/user-attachments/assets/d4c733b0-f741-4926-9373-40db7f4f837f)

![Screenshot 2024-10-25 190228](https://github.com/user-attachments/assets/4462df59-4a5b-47ab-b5ae-e07ef1de6f86)



## Error Handling

The system includes robust error handling for:
- Image loading failures
- Invalid input dimensions
- Training issues
- Memory constraints

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, and suggest features.


If you use this code in your research, please cite:
```
[Add citation information here]
```

For questions and support, please open an issue in the repository.

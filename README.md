# FCNet for TLE Prediction

FCNet is a fully Convolutional Neural Network designed for classfication of TLE sampels from 3D MRI data. 

## Project Structure
- **`models/`**: Contains model architectures and networks.
- **`visualization_techniques/`**: Contains functions to create saliency maps.
- **`config.py`**: Configuration settings for training and evaluation.
- **`trainer.py`**: Training loop and logging functions.
- **`main.py`**: Main entry point for running experiments.
- **`data_loader.py`**: Data loading and preprocessing scripts.
- **`utils.py/`**: Utility functions and helper scripts for data handling, visualization, and performance metrics.

## Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/rhassanzadeh/FCNet.git
   cd FCNet

## Example Command
1. **To train a model, use the following command:**:
   ```bash
   python main.py --is_train True
2. **To test a model, use the following command:**:
   ```bash
   python main.py --is_train False


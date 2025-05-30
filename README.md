# Multidisciplinary-Project-242: Bok Choy Disease Detection AI Project

This project is a part of my *Multidisciplinary Project* in semester 242 which aims to develop AI models for detecting diseases in bok choy plants. The project leverages deep learning techniques and distributed training to ensure scalability and efficiency. It is designed to be user-friendly and adaptable for various research and production environments.

## Dataset
The dataset is composed of two public datasets and converted to YOLO format. Links to the original author and datasets:
- Author: https://universe.roboflow.com/hydromac
- Datasets: 
  1. https://universe.roboflow.com/hydromac/bok-choy-healthy
  2. https://universe.roboflow.com/hydromac/bok-choy-diseased

The dataset is available on Kaggle at: [Bok Choy Disease Detection](https://www.kaggle.com/datasets/nguynhcan/bok-choy-disease-detection-yolo-format)

## Features

- **Distributed Data Parallel (DDP) Support**: The project includes DDP-friendly utilities, enabling efficient multi-GPU training. The [`ddp_init`](ddp.py) function in [ddp.py](ddp.py) simplifies the setup for distributed training.

- **CLI Training Script**: A command-line interface (CLI) script ([train.py](train.py)) allows users to train models with customizable arguments. This makes it easy to experiment with different configurations without modifying the code.

- **CLI Inferencing Script**: A command-line interface (CLI) script ([inference.py](inference.py)) allows users to inference the model with customizable arguments. 

- **Custom Model Architecture**: The project includes a custom model architecture with a specialized classification head, [`RetinaClassificationHeadDropout`](model.py), designed for robust disease detection.

- **Utility Functions**: Various helper functions in [utils.py](utils.py) streamline data processing, logging, and other tasks.

## File Overview

- **[ddp.py](ddp.py)**: Contains utilities for initializing Distributed Data Parallel (DDP) training.
- **[model.py](model.py)**: Implements the AI model architecture, including the classification head.
- **[train.py](train.py)**: CLI script for training the model with configurable arguments.
- **[utils.py](utils.py)**: Provides utility functions for data handling and other operations.
- **[data.py](data.py)**: Handles data loading and preprocessing for training and evaluation.
- **[engine.py](engine.py)**: Contains the training and evaluation loop logic.
- **[inference.py](inference.py)**: CLI script for inferencing the model with configurable arguments. 

## How to Use

1. **Install Dependencies**: Ensure you have Python and the required libraries installed. Use the following command to install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. **Train the Model**: Run the training script with desired arguments. For example:
    ```sh
    python train.py --epochs 50 --batch-size 32 --lr0 0.001
    ```

3. **Inferencing**: run the inference script with desired argument. For example:
    ```sh
    python inference.py --input-img ./sample_imgs/diseased1.jpg --threshold 0.2 --output_dir ./out_imgs --checkpoint-path ./best.pth
    ```

4. **Distributed Training**: Use the DDP setup for multi-GPU training. Refer to the [`ddp_init`](ddp.py) function for configuration details.

## WandB Logging (Optional)
If you want to track the train process with [Weights & Biases](https://wandb.ai/), enabled it with `--enable-logger 1` when run the script. For detail configuration, see the [`wandb_init`](utils.py) function in [utils.py](utils.py). 

Set your API key via environment variable or Kaggle secrets:
    ```python
    export WANDB_API_KEY=<your_api_key>
    ```

Or in Kaggle, add it via UserSecretsClient with key WANDB\_API\_KEY.

## License
This project is licensed under the LICENSE file.

## Contact
For questions or collaboration, please contact me. 

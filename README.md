# Crop-Weed Detection: YOLOv11 Baseline Performance and Ablation Studies

## Team CropGuard
Aye Khin Khin Hpone (Yolanda Lim-125970) | Julianna Godziszewska (126128) | Mir Ali Naqi Talpur (125001)

## Computer Vision

This research project evaluates YOLOv11 baseline performance for crop-weed detection and conducts ablation studies to assess the effectiveness of various enhancements. We systematically compare the baseline model against modifications including attention mechanisms (SimAM) and advanced loss functions (Varifocal Loss), demonstrating that the baseline YOLOv11 achieves optimal performance for this agricultural dataset.

## Project Overview

Crop-weed detection is a critical application in precision agriculture, enabling automated herbicide application and reducing environmental impact. This project explores multiple approaches to improve detection accuracy through:

- **Intelligent Data Augmentation**: Novel undersampling techniques for class-imbalanced datasets
- **Attention Mechanisms**: Integration of SimAM (Simple, Parameter-free Attention Module)
- **Advanced Loss Functions**: Implementation of Varifocal Loss for improved detection
- **Comprehensive Ablation Studies**: Systematic evaluation of each component's contribution

## Key Features

### 1. Data Augmentation Pipeline (`01_data_augmentation.ipynb`)
- Edge-based crop removal algorithm preserving ecological context
- Safety-first approach prioritizing weed instance preservation
- Iterative refinement maintaining spatial characteristics
- Statistical validation of augmentation effectiveness

### 2. Data Preprocessing & Ablation Study (`02_data_preprocessing_ablation.ipynb`)
- Comprehensive preprocessing pipeline optimization
- Comparative analysis of different preprocessing techniques
- Performance metrics evaluation across various preprocessing strategies

### 3. YOLOv11 with SimAM Integration (`03_yolov11_SimAM_ablation.ipynb`)
- Implementation of Simple, Parameter-free Attention Module (SimAM)
- Ablation study comparing baseline vs. attention-enhanced models
- Performance analysis on crop-weed detection tasks

### 4. Varifocal Loss Implementation (`04_yolov11_varifocal_loss_ablation.ipynb`)
- Advanced loss function for addressing class imbalance
- Comparative study with standard loss functions
- Impact assessment on detection accuracy

### 5. Model Inference & Visualization (`05_model_inference_and_visualization.ipynb`)
- Real-time inference pipeline
- Advanced visualization techniques for detection results
- Performance benchmarking and analysis

## Technical Stack

- **Deep Learning Framework**: PyTorch 2.9.0+
- **Object Detection**: YOLOv11 (Ultralytics 8.3.0+)
- **Computer Vision**: OpenCV 4.12.0+
- **Data Processing**: NumPy, Pandas, Polars
- **Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebooks

## Dataset

**Source**: [Mendeley Data - Crop-Weed Detection Dataset](https://data.mendeley.com/datasets/mthv4ppwyw/2)

The dataset contains annotated images of crops and weeds in agricultural settings, providing ground truth for object detection tasks.

## Installation

### Prerequisites
- Python 3.8+
- CUDA 12.x (for GPU training) or CPU-only setup

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Crop-Guard/crop-weed-detection-v3.git
   cd crop-weed-detection-v3
   ```

2. **Install PyTorch** (choose based on your setup):
   
   **For GPU with CUDA 12.x**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   
   **For CPU-only**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install remaining dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

## Quick Start

**Deployment Options**: This project can be run either on **puffer.cs.ait.ac.th** (AIT's GPU server) or on your own device with GPU support.

1. **Data Augmentation**: Start with `01_data_augmentation.ipynb` to prepare your dataset
2. **Preprocessing**: Run `02_data_preprocessing_ablation.ipynb` for data preprocessing
3. **Model Training**: Execute the model-specific notebooks:
   - `03_yolov11_SimAM_ablation.ipynb` for attention mechanism experiments
   - `04_yolov11_varifocal_loss_ablation.ipynb` for loss function studies
4. **Inference**: Use `05_model_inference_and_visualization-puffer.ipynb` for model evaluation

## Research Contributions

### Novel Attention Mechanism Integration
- **SimAM Implementation**: Parameter-free attention module that enhances feature representation without additional computational overhead
- **Architectural Innovation**: Seamless integration with YOLOv11 backbone for improved detection accuracy

### Advanced Data Augmentation Strategy
- **Edge-based Undersampling**: Novel approach to address class imbalance while preserving spatial context
- **Ecological Awareness**: Augmentation techniques that maintain realistic agricultural scenarios

### Comprehensive Ablation Studies
- **Component Analysis**: Systematic evaluation of each improvement's contribution
- **Performance Benchmarking**: Detailed comparison across multiple metrics and scenarios

## Results & Performance

Our experiments demonstrate significant improvements in crop-weed detection accuracy through:
- Enhanced detection of minority weed classes
- Improved spatial localization accuracy
- Reduced false positive rates in complex agricultural scenes
- Robust performance across diverse lighting and weather conditions

## Contributing

We welcome contributions to improve the project! Please feel free to:
- Report issues and bugs
- Suggest new features or improvements
- Submit pull requests with enhancements

## License

This project is open source and available under the MIT License.

## References

- **Dataset**: Crop-Weed Detection Dataset, Mendeley Data
- **YOLOv11**: Ultralytics YOLOv11 Object Detection
- **SimAM**: Simple, Parameter-Free Attention Module for Convolutional Neural Networks
- **Varifocal Loss**: Varifocal Loss for Dense Object Detection

## Contact

For questions, collaborations, or support, please reach out to Team CropGuard members:
- Aye Khin Khin Hpone (Yolanda Lim) - ID: 125970
- Julianna Godziszewska - ID: 126128
- Mir Ali Naqi Talpur - ID: 125001

---


*This project is part of ongoing research in computer vision applications for precision agriculture and sustainable farming practices.*

Berikut adalah contoh README untuk proyek Anda:

# CNNFormer: Transformer-based CNN for Image Classification

## Introduction
CNNFormer is a hybrid model that integrates Convolutional Neural Networks (CNN) and Transformers for the task of image classification. This model leverages the strengths of CNNs in feature extraction and the power of Transformers in contextual analysis. In this project, we utilize CNNFormer to classify images of two distinct art styles: Pop Art and Primitivism.

## Dataset
The dataset consists of images representing two art styles:
- **Pop Art**: Known for its bold colors, and use of commercial and popular imagery.
- **Primitivism**: Characterized by its inspiration from non-Western or prehistoric art, often using simple, geometric forms.

The dataset is divided into two folders:
- `dataset/Jenis Art/PopArt/`: Contains images of Pop Art.
- `dataset/Jenis Art/Primitivism/`: Contains images of Primitivism.

## Model Architecture
### CNN Component
The CNN component is responsible for extracting spatial features from the input images. It consists of multiple convolutional layers, followed by activation functions (ReLU) and max-pooling layers to reduce the spatial dimensions and focus on the most important features.

### Transformer Component
The Transformer component processes the extracted features to capture long-range dependencies and contextual information. It consists of Multi-Head Attention (MHA) mechanisms and feed-forward networks. Layer normalization is applied to stabilize the training.

### Combined Architecture
1. **Convolutional Layers**: Extract features from input images.
2. **Transformer Layers**: Analyze the contextual relationships between the extracted features.
3. **Fully Connected Layer**: Classify the processed features into the respective art style categories.

## Training and Evaluation
### Training
The model is trained using the Adam optimizer with a learning rate of 0.0001. Cross-Entropy Loss is used as the loss function. The training process involves:
- Loading and preprocessing the images.
- Feeding the images through the CNNFormer model.
- Computing the loss and updating the model parameters.

### Evaluation
The model's performance is evaluated on a separate test dataset. Key metrics include:
- **Accuracy**: The percentage of correctly classified images.
- **Confusion Matrix**: Visual representation of the model's performance in distinguishing between Pop Art and Primitivism.
- **ROC AUC Score**: Measure of the model's ability to distinguish between the two classes.

## Results
The model achieves high accuracy in classifying images of Pop Art and Primitivism. Detailed results, including accuracy, confusion matrix, and ROC AUC score, are presented in the evaluation section of the code.

## Contributing

Rahma Setiani 21102304
- Implementasi dan Perancangan Strukur Folder Program
- Training dan Evaluasi Model Program

Novi Ramadani 21102033
- Implementasi dan Evaluasi Program  
- Membuat dan Evaluasi Dataset Program 

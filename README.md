# DEEP-LEARNING-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: BHARATH K

INTERN ID: CT04DF229

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTHOSH

## TASK DESCRIPTION

This deep learning project focuses on implementing an end-to-end image classification model using a Convolutional Neural Network (CNN) with TensorFlow and Keras. The objective of the project is to design a 
functional deep learning pipeline capable of classifying handwritten digits from the MNIST dataset. This task involves loading and preprocessing image data, designing a CNN architecture, training the model, 
evaluating its performance, and visualizing key performance metrics such as accuracy and loss over training epochs.

The MNIST dataset is a standard benchmark in machine learning, consisting of 70,000 grayscale images of handwritten digits (0–9), with 60,000 used for training and 10,000 for testing. Each image is 28x28 
pixels in size and represents a single digit. The dataset is ideal for beginners and intermediate practitioners to develop and validate image classification models using deep learning techniques.

they are pixel values in  first stage of the project involves data loading and preprocessing. The dataset is loaded using TensorFlow's built-in utilities, and the e normalized to the range [0, 1] to improve 
model convergence. Additionally, since the CNN expects input with a channel dimension, the images are reshaped to (28, 28, 1) to indicate single-channel (grayscale) images.

The second stage is the model building phase, where a Sequential CNN model is constructed. The model begins with a 2D convolutional layer followed by a max pooling layer, repeated to extract more abstract
features. The convolutional layers use ReLU activation to introduce non-linearity, which enables the model to learn complex patterns. After feature extraction, the model is flattened and passed through
dense layers, ending with a softmax activation function that outputs probabilities for each of the 10 digit classes.


The third stage is model training, where the model is compiled using the Adam optimizer and trained using the sparse categorical crossentropy loss function, which is appropriate for integer-labeled
multi-class classification. The model is trained over five epochs with a batch size of 64, and 20% of the training data is used as a validation set. The accuracy and loss for both training and
validation data are recorded for analysis.

After training, the model is evaluated on the test dataset to measure generalization performance. The accuracy metric provides an overall measure of how well the model predicts unseen data. Finally,
the model’s performance is visualized using Matplotlib by plotting training and validation accuracy and loss over epochs. These plots help in diagnosing underfitting or overfitting and provide insights
into model behavior during learning.

In conclusion, this deep learning project effectively demonstrates the full workflow of image classification using CNNs with TensorFlow. It covers all essential components of a real-world machine learning
pipeline, from data preprocessing and model design to evaluation and visualization. The implementation is designed to run seamlessly in a Python environment such as Visual Studio Code, ensuring easy 
reproducibility and usability. The project not only strengthens the understanding of CNN architectures but also emphasizes the importance of data preparation and performance monitoring in deep learning
applications.

### OUTPUT: 

![Image](https://github.com/user-attachments/assets/7174887a-458b-4f23-8960-490551a88cc1)

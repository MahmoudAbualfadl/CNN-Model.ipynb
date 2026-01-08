# CNN-Model.ipynb
# CNN-based Rotation Angle Estimation

## Overview
This project implements a Convolutional Neural Network (CNN) to predict the rotation angles of MNIST images. The images are augmented by rotating them randomly, with the rotation angle as the target variable for this regression task. For comparison, a Feed-Forward Artificial Neural Network (ANN) is also trained on the same dataset. The performance of both models is evaluated using **Mean Absolute Error (MAE)** and **Loss**, with the goal of identifying which model is more effective for rotation angle estimation.

## Training Details

- **Optimizer**: Adam optimizer, a popular optimization algorithm that adapts the learning rate during training.
- **Loss Function**: Mean Squared Error (MSE) for regression tasks, which is used to measure the difference between predicted and actual rotation angles.
- **Metric**: Mean Absolute Error (MAE) is tracked during training to evaluate the average prediction error of the models.
- **Epochs**: The models are trained for 20 epochs with a batch size of 64.
- **Validation**: 20% of the data is used as validation data to evaluate model performance during training.

## Results

### Training History

- **Model Loss**: The loss decreases and stabilizes over time, indicating that the model is learning effectively.
- **Mean Absolute Error (MAE)**: The MAE values show that the model’s predictions are very close to the actual rotation angles.

### Observations

- The **CNN model** achieves significantly lower test loss and MAE compared to the **ANN model**, demonstrating the advantage of convolutional layers in handling image-based regression tasks.

### Next Steps

- **Increase Epochs**: Further training might improve the model’s performance if validation loss and MAE are still decreasing.
- **Model Tuning**: Adjusting model complexity, such as adding more layers or changing activation functions, could improve results if overfitting or underfitting occurs.

## Model Comparison

| Model | Test Loss | Test MAE |
|-------|-----------|----------|
| CNN   | 8.75      | 2.00     |
| ANN   | 34.21     | 5.78     |

## Files

- **train_images.npy**: Training dataset of MNIST images (augmented with random rotations).
- **train_angles.npy**: Labels corresponding to the rotation angles for the training set.
- **test_images.npy**: Test dataset of MNIST images (augmented with random rotations).
- **test_angles.npy**: Labels corresponding to the rotation angles for the test set.
- **cnn_model.h5**: The trained CNN model saved in H5 format.
- **ann_model.h5**: The trained ANN model saved in H5 format.

## Challenges and Solutions

- **Overfitting**: The CNN model exhibited signs of overfitting. This was addressed by adding **dropout layers** within the fully connected layers to regularize the model and improve its generalization.
  
- **Model Performance**: The **ANN model** underperformed compared to the CNN model due to the lack of convolutional layers. This experiment highlights the importance of convolutional layers for tasks involving spatial relationships in images, such as rotation angle estimation.

## Conclusion

The CNN model outperforms the ANN model in predicting rotation angles with a significantly lower error. This demonstrates that CNNs are more effective for image-based regression tasks, particularly those that involve spatial relationships such as rotation angle estimation.


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your dataset (assumed to be already preprocessed and stored in .npy files)
train_images = np.load('train_images.npy')
train_angles = np.load('train_angles.npy')
test_images = np.load('test_images.npy')
test_angles = np.load('test_angles.npy')

# Normalize images to range [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_angles, test_size=0.2, random_state=42)

# CNN Model Architecture
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the CNN model
history = cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64)

# Evaluate the CNN model on test data
cnn_test_loss, cnn_test_mae = cnn_model.evaluate(test_images, test_angles)

# ANN Model Architecture (for comparison)
ann_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])

# Compile the ANN model
ann_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the ANN model
ann_history = ann_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64)

# Evaluate the ANN model on test data
ann_test_loss, ann_test_mae = ann_model.evaluate(test_images, test_angles)

# Plot training history for CNN and ANN
plt.figure(figsize=(12, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='CNN Training Loss')
plt.plot(history.history['val_loss'], label='CNN Validation Loss')
plt.plot(ann_history.history['loss'], label='ANN Training Loss', linestyle='--')
plt.plot(ann_history.history['val_loss'], label='ANN Validation Loss', linestyle='--')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='CNN Training MAE')
plt.plot(history.history['val_mae'], label='CNN Validation MAE')
plt.plot(ann_history.history['mae'], label='ANN Training MAE', linestyle='--')
plt.plot(ann_history.history['val_mae'], label='ANN Validation MAE', linestyle='--')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()

# Comparison of results
print("Comparison of Test Results:")
print(f"CNN Test Loss: {cnn_test_loss:.2f}, CNN Test MAE: {cnn_test_mae:.2f}")
print(f"ANN Test Loss: {ann_test_loss:.2f}, ANN Test MAE: {ann_test_mae:.2f}")

# Save models
cnn_model.save('cnn_model.h5')
ann_model.save('ann_model.h5')

  <div>
  <img src="https://github.com/user-attachments/assets/59bcb958-4497-4d99-b3d3-851466e6dc4c">
</div>

  <div>
  <img src="https://github.com/user-attachments/assets/218c38e6-2133-47fe-bf4b-5cea590a8c02">
</div>



import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Sequential
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. Dataset Loading (Using CIFAR-10) [cite: 23, 24]
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 2. Image Preprocessing: Normalization [cite: 25, 27]
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 3. Model Building with Dropout and Batch Normalization [cite: 29, 31, 32]
model = Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4. Model Architecture Summary [cite: 44]
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Model Training (Set to 10-20 epochs per project guide) [cite: 33, 35]
history = model.fit(train_images, train_labels, epochs=15, 
                    validation_data=(test_images, test_labels))

# Save the model for your Flask app
model.save('model.h5')

# 6. Visualize Accuracy and Loss Curves [cite: 15, 36, 45]
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

# 7. Model Evaluation: Confusion Matrix [cite: 37, 38]
predictions = model.predict(test_images)
pred_labels = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_labels, pred_labels)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
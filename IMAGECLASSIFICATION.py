import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set paths (adjust based on your Google Drive structure)
dataset_path = '/content/drive/My Drive/vehicle_dataset/'  # Path to the dataset in Google Drive
img_path = '/content/drive/My Drive/test_vehicle_image.jpg'  # Path to the test image for prediction
model_save_path = '/content/drive/My Drive/vehicle_classification_model.h5'  # Path to save the trained model

# Image properties
img_height, img_width = 128, 128
batch_size = 32
epochs = 10

# Data augmentation and image preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,                # Rescale pixel values from [0, 255] to [0, 1]
    rotation_range=10,             # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,         # Randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,        # Randomly shift images vertically (fraction of total height)
    brightness_range=[0.8, 1.2],   # Randomly change brightness
    horizontal_flip=True,          # Randomly flip images horizontally
    validation_split=0.2           # 80% training, 20% validation
)

# Load the training data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Use training subset
)

# Load the validation data
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Use validation subset
)

# Define the CNN model
num_classes = len(train_generator.class_indices)  # Dynamically get the number of classes
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Dynamically set the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Save the trained model to Google Drive
model.save(model_save_path)
print(f"Model training complete and saved as {model_save_path}")

# Prediction on a new image
# Load the trained model
model = tf.keras.models.load_model(model_save_path)

# Define class indices dynamically
class_indices = {v: k for k, v in train_generator.class_indices.items()}  # Reverse the class indices

# Load and preprocess the image for prediction
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Get the corresponding vehicle type
predicted_vehicle = class_indices[predicted_class]

print(f"Predicted vehicle type: {predicted_vehicle}")
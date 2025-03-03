import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset directories
train_dir = "dataset/train"
test_dir = "dataset/test"

# Ensure model directory exists
model_dir = os.path.join(os.getcwd(), "model")
os.makedirs(model_dir, exist_ok=True)

# Load images using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode="categorical"
)

test_generator = datagen.flow_from_directory(
    test_dir, target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode="categorical"
)

# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=20, validation_data=test_generator)

# Save the model
model_path = os.path.join(model_dir, "facial_emotion_model.h5")
model.save(model_path)

print(f"✅ Model saved at models")

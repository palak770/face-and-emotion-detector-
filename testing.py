import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
# Make sure these match the settings you used for training
MODEL_PATH = 'my_custom_emotion_model.h5' # The model you want to test
TEST_DIR = 'test'
IMAGE_SIZE = (96, 96) # Use (128, 128) if you trained the VGG16 model
BATCH_SIZE = 32

# 1. Load the trained model
print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# 2. Load the test data (without any augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # IMPORTANT: Do not shuffle the data for evaluation
)

# 3. Get the final accuracy score
print("\n--- Evaluating Model Performance ---")
loss, accuracy = model.evaluate(test_generator)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
print(f"Final Test Loss: {loss:.4f}")

# 4. Generate a detailed Classification Report
print("\n--- Generating Classification Report ---")
# Get the model's predictions for the entire test set
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

# Get the true labels
y_true = test_generator.classes

# Get the class names
class_labels = list(test_generator.class_indices.keys())

# Print the report
print(classification_report(y_true, y_pred, target_names=class_labels))

# 5. Generate and display a Confusion Matrix
print("\n--- Generating Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

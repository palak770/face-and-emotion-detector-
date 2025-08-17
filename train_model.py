import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# --- CHANGE 1: Import VGG16 ---
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- CHANGE 2: VGG16 works well with larger images ---
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
TRAIN_DIR = 'train'
TEST_DIR = 'test'

# --- Data Generators (updated with new image size) ---
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.1)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# --- CHANGE 3: Build the model using VGG16 ---
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x) # VGG16 uses a Flatten layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile and train the model (use a smaller learning rate for fine-tuning)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print("--- Starting VGG16 Model Training ---")
model.fit(train_generator, epochs=15, validation_data=validation_generator)
model.save('my_vgg16_emotion_model.h5')
print("--- Training Complete! Model saved as my_vgg16_emotion_model.h5 ---")

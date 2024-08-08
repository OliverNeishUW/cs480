import numpy as np
import pandas as pd 
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate

target_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
# File paths would need to be changed to run on marker's system
train_df = pd.read_csv('C:/Users/olive/Downloads/cs-480-2024-spring/data/train.csv')
test_df = pd.read_csv('C:/Users/olive/Downloads/cs-480-2024-spring/data/test.csv')

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess images
def load_images(image_ids, img_dir):
    images = []
    i = 0
    for img_id in image_ids:
        img_path = os.path.join(img_dir, f"{img_id}.jpeg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = np.array(img) / 255.0
            images.append(img)
    return np.array(images)

# File paths would need to be modified to run on marker's system
train_images = load_images(train_df['id'], 'C:/Users/olive/Downloads/cs-480-2024-spring/data/train_images')
test_images = load_images(test_df['id'], 'C:/Users/olive/Downloads/cs-480-2024-spring/data/test_images')

ancillary_columns = train_df.columns[1:164]
# Normalize the ancillary data
scaler = StandardScaler()
train_ancillary = scaler.fit_transform(train_df[ancillary_columns])
test_ancillary = scaler.transform(test_df[ancillary_columns])

# Define the image input
image_input = Input(shape=(128, 128, 3))
model = tf.keras.applications.ResNet50(include_top=False, input_tensor=image_input, pooling='avg')

# Get the output
model_output = model.output
x = Flatten()(model_output)

# Define the ancillary data input
ancillary_input = Input(shape=(163,))
y = Dense(128, activation='relu')(ancillary_input)

# Combine outputs
combined = Concatenate()([x, y])
z = Dense(128, activation='relu')(combined)
z = Dense(6)(z)

model = Model(inputs=[image_input, ancillary_input], outputs=z)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    [train_images, train_ancillary],
    train_df[target_columns],
    epochs=30,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# Commented out for efficiency
# loss, mae = model.evaluate([test_images, test_ancillary], test_df[target_columns])
# print(f"Test MAE: {mae}")

print("Predicting now...")
predictions = model.predict([test_images, test_ancillary])

print("Saving now...")
submission_columns = ['id', 'X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
submission_df = pd.DataFrame(predictions, columns=target_columns)
submission_df['id'] = test_df['id'].values
submission_df = submission_df[['id'] + target_columns]
submission_df.to_csv('C:/Users/olive/Downloads/submission.csv', index=False) # File needs to be modified

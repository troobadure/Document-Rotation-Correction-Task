import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn import preprocessing
from joblib import dump

images_path = 'data/rdocuments/rdocuments/'
csv_path = 'data/rdocuments/r-images.csv'

IMG_WIDTH, IMG_HEIGHT = 316, 316
BATCH_SIZE = 32


labels_df = pd.read_csv(csv_path)

scaler = preprocessing.StandardScaler()
labels_df['angle_scaled'] = scaler.fit_transform(labels_df['angle'].values.reshape(-1, 1))

y_train = labels_df['angle_scaled'].values


X_train = []

for file_name in labels_df['id']:
    image = cv2.imread(os.path.join(images_path, file_name), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # COMPUTING FFT
    f = cv2.dft(np.float32(image))
    fshift = np.fft.fftshift(f)
    f_abs = np.abs(fshift) + 1.0
    f_img = 20 * np.log(f_abs)
    
    X_train.append(f_img)

X_train = np.array(X_train)
X_train = X_train.reshape(len(X_train), IMG_HEIGHT, IMG_WIDTH, 1)
X_train /= 255


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)))

model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), 
                                   activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), 
                                   activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), 
                                   activation='relu', padding='same'))
model.add(tf.keras.layers.GlobalMaxPooling2D())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss=tf.keras.losses.mean_absolute_error,
                metrics=[tf.keras.metrics.mean_squared_error])


hist = model.fit(X_train,
                    y_train,
                    validation_split=0.2,
                    batch_size=BATCH_SIZE,
                    epochs=50,
                    verbose=2)

model.save('models/cnn_with_fft.h5')
dump(scaler, 'models/scaler.bin', compress=True)
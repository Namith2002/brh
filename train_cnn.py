import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

TRAINING_DATA_DIR = "training_data"

CLASS_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

def mapper(value):
    return CLASS_MAP[value]

def create_model(input_shape, num_classes):
    model = keras.models.Sequential([

        keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The third convolution
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The fifth convolution
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def get_dataset(img_shape):
    # load images from the training data directory
    dataset = []
    for label_dir in os.listdir(TRAINING_DATA_DIR):
        # iterating each item in the training data directory
        path = os.path.join(TRAINING_DATA_DIR, label_dir)
        if not os.path.isdir(path):
            continue
        # iterating threw each file in the sub directory
        for image_file in os.listdir(path):
            # loading each image
            img = cv2.imread(os.path.join(path, image_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_shape)
            
            # adding them to the dataset
            dataset.append([img, label_dir])
    
    return zip(*dataset)

def main():

    img_shape = (225, 225)
    dataset, labels = get_dataset(img_shape)

    # label encode the classes
    labels = list(map(mapper, labels))

    input_shape = (225, 225, 3)
    model = create_model(input_shape, len(CLASS_MAP))

    with tf.device('/device:GPU:0'):

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()

        # start training
        model.fit(np.array(dataset), np.array(labels), epochs=15)

        # save the model for later use
        model.save("modelnew.h5")


if __name__ == "__main__":
    main()

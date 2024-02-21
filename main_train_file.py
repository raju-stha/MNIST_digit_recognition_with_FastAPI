import sys
import tensorflow as tf
from keras import layers, models, datasets
import matplotlib.pyplot as plt
import numpy as np

# b_train = False

# Load MNIST dataset
(train_images, train_labels), (test_images,
                               test_labels) = datasets.mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


def train_or_predict(b_train):

    if b_train:
        # Build the model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3),
                          activation='relu',
                          input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(train_images, train_labels, epochs=5, batch_size=64)

        # Save the model
        model.save('mnist_model.h5')

        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)
    else:
        # Load the model
        model = models.load_model('mnist_model.h5')
        while True:

            # Get the input image index
            test_image_ind = int(
                input(
                    '\n #################################-------------------########### \n Enter the test image index to predict: '
                ))

            # Demo with input as image and output as prediction
            sample_image = test_images[int(test_image_ind)]
            sample_image = np.expand_dims(sample_image, axis=0)
            prediction = model.predict(sample_image, verbose=0)
            print('The prediction for the given image is:',
                  np.argmax(prediction))
            plt.imshow(test_images[int(test_image_ind)].reshape(28, 28),
                       cmap='gray')
            plt.title('The prediction for the given image is: ' +
                      str(np.argmax(prediction)))
            plt.show()
            print('Do you want to predict another image? (y/n)')
            if input() == 'n':
                break


# take the system arg input to decide whether to train the model or not
if __name__ == "__main__":
    sys_arg = sys.argv
    assert len(
        sys_arg
    ) > 1, 'Please provide system arg python initial_test.py train or python initial_test.py predict'
    if str(sys_arg[1]) == 'train':
        b_train = True
        print('Training the model')
        train_or_predict(b_train)
    else:
        b_train = False
        train_or_predict(b_train)

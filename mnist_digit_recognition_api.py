from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from keras import layers, models, datasets
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# Load your model
model = models.load_model('mnist_model.h5')

# Load MNIST dataset
(train_images,
 train_labels), (test_images,
                 test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255


# Define a Pydantic model for the input data
class InputData(BaseModel):
    input_integer: int
    # Add more features as needed


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow requests from all origins (you may want to restrict this in production)
    allow_credentials=True,
    allow_methods=["GET", "POST",
                   "OPTIONS"],  # Add OPTIONS to the allowed methods
    allow_headers=["*"],
)


# Function to generate plot as base64 encoded string
def generate_plot(test_image_ind):
    """
    Function to generate plot as base64 encoded string
    :param test_image_ind: Index of the test image
    :return: Base64 encoded string of the plot"""

    # Demo with input as image and output as prediction
    sample_image = test_images[int(test_image_ind)]
    sample_image = np.expand_dims(sample_image, axis=0)
    prediction = model.predict(sample_image, verbose=0)
    # print('The prediction for the given image is:', np.argmax(prediction))
    plt.imshow(test_images[int(test_image_ind)].reshape(28, 28), cmap='gray')
    plt.title('The prediction for the given image is: ' +
              str(np.argmax(prediction)))

    # Save plot to bytes IO
    plot_bytes = io.BytesIO()
    plt.savefig(plot_bytes, format='png')
    plot_bytes.seek(0)

    # Convert plot to base64 encoded string
    plot_base64 = base64.b64encode(plot_bytes.getvalue()).decode()

    plt.close()  # Close the plot to free memory

    return plot_base64, np.argmax(prediction)


# Define a route to handle POST requests
@app.post("/estimate/")
async def estimate(data: InputData):
    """
    Return the prediction and plot as base64 encoded string
    
    Args:
    - data: Input data as a Pydantic model
    Returns:
          - a dictionary containing the prediction and plot as base64 encoded string

    """
    # Convert input data to the format expected by the model
    input_feature = data.input_integer  # Add more features as needed
    # print("this has been called")
    # Generate plot
    plot_base64, prediction = generate_plot(input_feature)

    # Return the prediction and plot as base64 encoded string
    return {"prediction": int(prediction), "plot": plot_base64}


if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    pass

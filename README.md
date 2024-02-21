# MNIST Digit Recognition API
This repository contains a FastAPI-based web API for recognizing handwritten digits using a pre-trained deep learning model. It provides an HTML form for users to input a digit index from the MNIST dataset, and the API returns the predicted digit along with a plot of the corresponding image. The deep learning model used in this project is trained on the MNIST dataset.


# Contents:
`mnist_digit_recognition_api.py`: Python script containing the FastAPI application code.
`mnist_digit_recognition_html.html`: HTML file for the user interface.
`Dockerfile`: Dockerfile for containerizing the application.
`requirements.txt`: File containing the Python dependencies required for running the application.
`mnist_model.h5`: Saved keras model trained on the MNIST dataset.

# Usage:
### A) Running the API locally:

1) Clone the repository.
2) Install the required Python dependencies using pip install -r requirements.txt.
3) Run the FastAPI application using uvicorn mnist_digit_recognition_api:app --host 0.0.0.0 --port 8000.
4) Access the API with `mnist_digit_recognition_html.html`. 

### B) Using the Docker image:

1) Build the Docker image using `docker build -t mnist-digit-recognition .`.
2) Run the Docker container using `docker run -p 8000:8000 --rm mnist-digit-recognition`.
3) Access the API with `mnist_digit_recognition_html.html`. 

# Dependencies:
Please check the `requirements.txt` file.

# Contact
Feel free to contribute to this project by submitting bug reports, feature requests, or pull requests. Happy digit recognition!

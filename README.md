# AI-Based Battery's State of Charge Estimator

This project aims to develop an AI model to estimate the state of charge (SOC) of lithium-ion batteries under different temperatures and operational modes based on measured voltage data.

## Overview

The project consists of two main components:

1. **OCV Model**: This component focuses on training an AI model using the open circuit voltage (OCV) data. The OCV data includes battery voltage, temperature, and mode of operation. The model is trained using LSTM (Long Short-Term Memory) neural network architecture to predict the SOC based on the input features.

2. **DDPT Model**: This component involves training another AI model using the dynamic discharge pulse test (DDPT) data. Similar to the OCV model, the DDPT data includes voltage, temperature, current, and mode of operation. The LSTM neural network is employed to predict the SOC based on these input features.

## Usage

### OCV Model

To train the OCV model:

1. Ensure you have installed the required dependencies listed in `requirements.txt`.
2. Run the `ocv_training.py` script to preprocess the OCV data, train the model, and save the trained model.

### DDPT Model

To train the DDPT model:

1. Ensure you have installed the required dependencies listed in `requirements.txt`.
2. Run the `ddpt_training.py` script to preprocess the DDPT data, train the model, and save the trained model.

### Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- TensorFlow

## Data

The dataset used in this project can be found [here](https://docs.google.com/spreadsheets/d/1UFEsXchsAojxhX9SO6gzS_jXSO51JBRs/edit?usp=drive_web&ouid=110622254758429064134&rtpof=true). It includes OCV and DDPT data for different temperatures and operational modes.

## Author

This project is authored by [Your Name].

## License

This project is licensed under the [MIT License](LICENSE).

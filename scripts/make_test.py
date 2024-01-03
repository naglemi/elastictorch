import sys
import torch
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from elastictorch import ElasticNet, PyTorchOptimizerTrainer

def run_model_on_device(device='cpu'):
    # Start timing
    start_time = time.time()

    # Set the device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Load the data
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Preprocess the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the data to PyTorch tensors and move them to the specified device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Create an instance of the ElasticNet model and move it to the specified device
    model = ElasticNet(n_features=X_train.shape[1]).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Create an instance of the trainer
    trainer = PyTorchOptimizerTrainer(model, optimizer)

    # Train the model
    trainer.train(X_train, y_train, epochs=1000)

    # Use the trained model to make predictions on the test data
    predictions = model(X_test)

    # Move predictions back to CPU for further operations like saving to CSV
    predictions = predictions.cpu().detach().numpy()

    # End timing
    end_time = time.time()

    # Calculate and print the duration
    duration = end_time - start_time
    print(f"Total time taken: {duration} seconds")

    # Save the predictions to a CSV file
    pd.DataFrame(predictions).to_csv("predictions.csv", index=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python model_benchmark.py [cpu/gpu]")
    else:
        device_arg = sys.argv[1].lower()
        if device_arg not in ['cpu', 'cuda']:
            print("Invalid argument. Please specify 'cpu' or 'cuda'.")
        else:
            run_model_on_device(device_arg)

import torch

class PyTorchOptimizerTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, X_train, y_train, epochs=1000):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.model(X_train)
            loss = self.model.loss(predictions, y_train)
            loss.backward()
            self.optimizer.step()

class ManualOptimizerTrainer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def train(self, X_train, y_train, epochs=1000):
        for epoch in range(epochs):
            predictions = self.model(X_train)
            loss = self.model.loss(predictions, y_train)
            loss.backward()

            with torch.no_grad():
                for param in self.model.parameters():
                    param -= self.learning_rate * param.grad

            self.model.zero_grad()

class Trainer(PyTorchOptimizerTrainer, ManualOptimizerTrainer):
    def __init__(self, model, optimizer=None, learning_rate=None):
        if optimizer is not None:
            PyTorchOptimizerTrainer.__init__(self, model, optimizer)
        elif learning_rate is not None:
            ManualOptimizerTrainer.__init__(self, model, learning_rate)
        else:
            raise ValueError("Must provide either an optimizer or a learning rate")
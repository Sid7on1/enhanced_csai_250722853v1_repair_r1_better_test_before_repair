import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
MODEL_PATH = 'models/agent.pth'
DATA_PATH = 'data/agent_data.csv'
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# Custom exception classes
class TrainingError(Exception):
    """Custom exception class for training-related errors."""
    pass

class DataLoadingError(Exception):
    """Custom exception class for data loading errors."""
    pass

# Helper classes and utilities
class AgentDataset(Dataset):
    """Custom dataset class for loading and preprocessing the agent data."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.data_path)
            if any(data.isnull()):
                raise ValueError("Missing data values in the dataset.")
            return data
        except FileNotFoundError:
            raise DataLoadingError(f"Data file not found at path: {self.data_path}")
        except pd.errors.EmptyDataError:
            raise DataLoadingError("Data file is empty.")
        except pd.errors.ParserError:
            raise DataLoadingError("Data file is improperly formatted and cannot be parsed.")
        except ValueError as e:
            raise DataLoadingError(str(e))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        return {'input': torch.tensor(row['input']),
                'target': torch.tensor(row['target'])}

class AgentModel(nn.Module):
    """Custom neural network model for the agent."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(AgentModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

# Main class with methods
class AgentTrainer:
    """Agent training pipeline."""

    def __init__(self, data_path: str, model_path: str, batch_size: int, num_epochs: int, learning_rate: float):
        self.data_path = data_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def load_data(self) -> DataLoader:
        """Load and preprocess the training data."""
        try:
            dataset = AgentDataset(self.data_path)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return data_loader
        except DataLoadingError as e:
            logger.error(f"Error loading data: {e}")
            raise TrainingError("Failed to load data.")

    def build_model(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Build and compile the agent model."""
        self.model = AgentModel(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def train_model(self, data_loader: DataLoader) -> None:
        """Train the agent model using the provided data loader."""
        self.model.train()
        total_loss = 0.0
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch in data_loader:
                inputs, targets = batch['input'].to(self.device), batch['target'].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            epoch_loss /= len(data_loader)
            total_loss += epoch_loss
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}: Loss = {epoch_loss:.4f}")
        avg_loss = total_loss / self.num_epochs
        logger.info(f"Average loss across epochs: {avg_loss:.4f}")

    def save_model(self) -> None:
        """Save the trained model to disk."""
        if not self.model:
            raise TrainingError("Model has not been built or trained yet.")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Model saved successfully at path: {self.model_path}")

    def train(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """End-to-end training pipeline."""
        data_loader = self.load_data()
        self.build_model(input_size, hidden_size, output_size)
        self.train_model(data_loader)
        self.save_model()

def main():
    trainer = AgentTrainer(data_path=DATA_PATH, model_path=MODEL_PATH, batch_size=BATCH_SIZE,
                          num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    trainer.train(input_size=10, hidden_size=64, output_size=1)

if __name__ == '__main__':
    main()
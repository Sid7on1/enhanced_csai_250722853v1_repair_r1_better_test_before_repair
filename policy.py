import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union

class PolicyNetwork:
    """
    Policy Network class for implementing the policy-based agent.

    This class provides an implementation of a policy network that can be used for
    training and inference in a reinforcement learning setting. It includes methods for
    defining the network architecture, performing forward passes, and calculating
    probabilities and actions based on the policy.

    ...

    Attributes
    ----------
    input_size : int
        The size of the input feature vector.
    output_size : int
        The number of possible actions that the agent can take.
    hidden_layers : List[int]
        A list of integers representing the number of neurons in each hidden layer.
    device : torch.device
        The device on which the tensors will be stored (CPU or GPU).
    model : torch.nn.Module
        The neural network model used for the policy.
    optimizer : torch.optim
        The optimizer used for updating the model weights during training.
    loss_fn : callable
        The loss function used for updating the model weights during training.

    Methods
    -------
    forward(inputs: torch.Tensor) -> torch.Tensor
        Perform a forward pass through the network.
    calculate_probabilities(state: np.ndarray) -> np.ndarray
        Calculate the probabilities of taking each action given a state.
    select_action(state: np.ndarray) -> Union[int, str]
        Select an action based on the policy and a given state.
    update_parameters(states: List[np.ndarray], actions: List[Union[int, str]], rewards: List[float]) -> None
        Update the model parameters based on the provided states, actions, and rewards.
    save_model(filepath: str) -> None
        Save the trained model to a file.
    load_model(filepath: str) -> None
        Load a pre-trained model from a file.

    """

    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int], device: str = 'cpu',
                 learning_rate: float = 0.001):
        """
        Initialize the PolicyNetwork class.

        Parameters
        ----------
        input_size : int
            The size of the input feature vector.
        output_size : int
            The number of possible actions that the agent can take.
        hidden_layers : List[int]
            A list of integers representing the number of neurons in each hidden layer.
        device : str, optional
            The device on which the tensors will be stored (CPU or GPU), by default 'cpu'.
        learning_rate : float, optional
            The learning rate for the optimizer, by default 0.001.

        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.device = torch.device(device)
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _build_model(self) -> torch.nn.Module:
        """
        Build the neural network model for the policy.

        Returns
        -------
        torch.nn.Module
            The constructed neural network model.

        """
        model = torch.nn.Sequential()
        model.add_module('input_layer', torch.nn.Linear(self.input_size, self.hidden_layers[0]))
        model.add_module('relu_1', torch.nn.ReLU())

        for i in range(1, len(self.hidden_layers)):
            model.add_module(f'hidden_layer_{i}', torch.nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
            model.add_module(f'relu_{i+1}', torch.nn.ReLU())

        model.add_module('output_layer', torch.nn.Linear(self.hidden_layers[-1], self.output_size))
        model.add_module('softmax', torch.nn.Softmax(dim=1))

        return model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            A tensor of input features of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            A tensor of output probabilities of shape (batch_size, output_size).

        """
        return self.model(inputs.to(self.device))

    def calculate_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate the probabilities of taking each action given a state.

        Parameters
        ----------
        state : np.ndarray
            An array of state features of shape (input_size,).

        Returns
        -------
        np.ndarray
            An array of probabilities for each action of shape (output_size,).

        """
        state_tensor = torch.from_numpy(state).float().to(self.device)
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        output = self.forward(state_tensor)
        probabilities = output.squeeze(0).detach().cpu().numpy()
        return probabilities

    def select_action(self, state: np.ndarray) -> Union[int, str]:
        """
        Select an action based on the policy and a given state.

        Parameters
        ----------
        state : np.ndarray
            An array of state features of shape (input_size,).

        Returns
        -------
        Union[int, str]
            The selected action, either as an integer index or a string representation.

        """
        probabilities = self.calculate_probabilities(state)
        action = np.random.choice(self.output_size, p=probabilities)
        return action

    def update_parameters(self, states: List[np.ndarray], actions: List[Union[int, str]], rewards: List[float]) -> None:
        """
        Update the model parameters based on the provided states, actions, and rewards.

        Parameters
        ----------
        states : List[np.ndarray]
            A list of state feature arrays of shape (input_size,).
        actions : List[Union[int, str]]
            A list of actions taken, either as integer indices or string representations.
        rewards : List[float]
            A list of corresponding rewards received after taking each action.

        """
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        policy_outputs = self.forward(states_tensor)
        loss = self.loss_fn(policy_outputs, actions_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.

        Parameters
        ----------
        filepath : str
            The path to the file where the model will be saved.

        """
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load a pre-trained model from a file.

        Parameters
        ----------
        filepath : str
            The path to the file from which the model will be loaded.

        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

class PolicyAgent:
    """
    Policy Agent class for implementing an agent that uses a policy network.

    This class provides an interface for interacting with the policy network and
    includes methods for training, inference, and evaluation.

    ...

    Attributes
    ----------
    policy : PolicyNetwork
        The policy network used by the agent.
    explorer : Explorer
        The explorer module used for exploration during training.
    env : Environment
        The environment in which the agent interacts.

    Methods
    -------
    train(num_episodes: int, exploration_rate: float) -> None
        Train the agent for a specified number of episodes.
    act(state: np.ndarray) -> Union[int, str]
        Select an action based on the current state.
    evaluate(num_episodes: int) -> float
        Evaluate the performance of the trained agent.

    """

    def __init__(self, policy: PolicyNetwork, explorer: Explorer, env: Environment):
        """
        Initialize the PolicyAgent class.

        Parameters
        ----------
        policy : PolicyNetwork
            The policy network used by the agent.
        explorer : Explorer
            The explorer module used for exploration during training.
        env : Environment
            The environment in which the agent interacts.

        """
        self.policy = policy
        self.explorer = explorer
        self.env = env

    def train(self, num_episodes: int, exploration_rate: float) -> None:
        """
        Train the agent for a specified number of episodes.

        Parameters
        ----------
        num_episodes : int
            The number of episodes to train the agent for.
        exploration_rate : float
            The rate of exploration during training.

        """
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_rewards = []

            while not done:
                action = self.select_action(state, exploration_rate)
                next_state, reward, done = self.env.step(action)
                self.policy.update_parameters(state, action, reward)
                state = next_state
                episode_rewards.append(reward)

            self.explorer.update_exploration(episode, episode_rewards)

    def select_action(self, state: np.ndarray, exploration_rate: float) -> Union[int, str]:
        """
        Select an action based on the current state and exploration rate.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.
        exploration_rate : float
            The rate of exploration.

        Returns
        -------
        Union[int, str]
            The selected action, either as an integer index or a string representation.

        """
        if np.random.uniform(0, 1) < exploration_rate:
            action = self.explorer.explore()
        else:
            action = self.policy.select_action(state)

        return action

    def evaluate(self, num_episodes: int) -> float:
        """
        Evaluate the performance of the trained agent.

        Parameters
        ----------
        num_episodes : int
            The number of episodes to evaluate the agent for.

        Returns
        -------
        float
            The average reward received during evaluation.

        """
        total_rewards = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_rewards = 0

            while not done:
                action = self.policy.select_action(state)
                state, reward, done = self.env.step(action)
                episode_rewards += reward

            total_rewards += episode_rewards

        return total_rewards / num_episodes

# Example usage
if __name__ == '__main__':
    # Instantiate the policy network, explorer, and environment
    policy_net = PolicyNetwork(input_size=4, output_size=3, hidden_layers=[64, 32])
    explorer = Explorer()  # Explorer class implementation
    env = Environment()  # Environment class implementation

    # Create the policy agent
    agent = PolicyAgent(policy_net, explorer, env)

    # Train the agent
    agent.train(num_episodes=1000, exploration_rate=0.1)

    # Evaluate the trained agent
    avg_reward = agent.evaluate(num_episodes=100)
    print(f"Average reward during evaluation: {avg_reward}")
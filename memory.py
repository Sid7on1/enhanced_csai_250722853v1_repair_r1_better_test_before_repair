import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from torch import Tensor
except ImportError as e:
    logger.error("Failed to import torch. Install torch package.")
    raise e

# Constants and configuration
MEMORY_SIZE = 10000
MIN_OBSERVATION_SIZE = 100
VELOCITY_THRESHOLD = 0.1

# Exception classes
class MemoryError(Exception):
    pass

class ExperienceReplayMemory:
    """
    Experience replay memory for storing and sampling transitions.
    """

    def __init__(self, memory_size: int = MEMORY_SIZE):
        """
        Initializes the experience replay memory.

        Args:
            memory_size (int): Maximum number of transitions to store.
        """
        self.memory_size = memory_size
        self.memory = []
        self.position = 0
        self._lock = False

    def remember(self, transition: Tuple[Tensor, Tensor, float, Tensor, bool]) -> None:
        """
        Stores a transition in memory.

        Args:
            transition (Tuple[Tensor, Tensor, float, Tensor, bool]):
                - state (Tensor): Current state observation.
                - action (Tensor): Action taken.
                - reward (float): Reward received.
                - next_state (Tensor): Next state observation.
                - done (bool): Whether the episode has terminated.

        Raises:
            MemoryError: If the memory is full.
        """
        if len(self.memory) < self.memory_size:
            self.memory.append(transition)
        else:
            raise MemoryError("Memory is full. Cannot store more transitions.")

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """
        Samples a batch of transitions for training.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Dict[str, Tensor]: Batched transitions with keys:
                - states (Tensor): Batched current states.
                - actions (Tensor): Batched actions.
                - rewards (Tensor): Batched rewards.
                - next_states (Tensor): Batched next states.
                - dones (Tensor): Batched episode termination flags.
        """
        if len(self.memory) < MIN_OBSERVATION_SIZE:
            raise MemoryError(
                "Memory does not have enough observations for sampling."
            )

        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return {
            "states": Tensor(states),
            "actions": Tensor(actions),
            "rewards": Tensor(rewards),
            "next_states": Tensor(next_states),
            "dones": Tensor(dones),
        }

    def __len__(self) -> int:
        """
        Returns the current number of stored transitions.

        Returns:
            int: Length of the memory.
        """
        return len(self.memory)

# Helper class for priority-based experience replay
class SumTree:
    """
    SumTree data structure for efficient priority-based experience replay.
    """

    def __init__(self, capacity: int):
        """
        Initializes the SumTree.

        Args:
            capacity (int): Maximum number of elements the SumTree can hold.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.current_pos = 0

    def _propagate(self, idx: int, change: float) -> None:
        """
        Propagates the change in priority upwards in the tree.

        Args:
            idx (int): Index of the element to update.
            change (float): Change in priority.
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> Tuple[int, float]:
        """
        Retrieves the element with the highest priority.

        Args:
            idx (int): Current index in the tree.
            s (float): Current priority sum.

        Returns:
            Tuple[int, float]: Index of the selected element and its priority.
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx, s

        if self.tree[left] > self.tree[right]:
            idx = left
        else:
            idx = right
            s += self.tree[left]

        return self._retrieve(idx, s)

    def add(self, priority: float, data: object) -> int:
        """
        Adds an element to the SumTree.

        Args:
            priority (float): Priority of the element.
            data (object): Data associated with the priority.

        Returns:
            int: Index of the added element.
        """
        idx = self.current_pos + self.capacity - 1
        self.data[self.current_pos] = data
        self.update(idx, priority)

        self.current_pos += 1
        if self.current_pos >= self.capacity:
            self.current_pos = 0

        return idx

    def update(self, idx: int, priority: float) -> None:
        """
        Updates the priority of an element in the SumTree.

        Args:
            idx (int): Index of the element to update.
            priority (float): New priority value.
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[object, float]:
        """
        Retrieves an element based on the priority sum.

        Args:
            s (float): Priority sum to retrieve the element for.

        Returns:
            Tuple[object, float]: Data associated with the element and its priority.
        """
        idx, s = self._retrieve(0, s)
        data = self.data[idx % self.capacity]
        priority = self.tree[idx]
        return data, priority

# Priority-based experience replay memory
class PrioritizedReplayMemory:
    """
    Experience replay memory with priority-based sampling.
    """

    def __init__(self, memory_size: int = MEMORY_SIZE):
        """
        Initializes the prioritized replay memory.

        Args:
            memory_size (int): Maximum number of transitions to store.
        """
        self.memory_size = memory_size
        self.sum_tree = SumTree(self.memory_size)
        self.beta = 0.6  # Controls the amount of prioritization
        self.beta_increment = 0.001  # Increment for beta
        self.absolute_error_upper = 1.  # Clip absolute error at this value

    def remember(self, transition: Tuple[Tensor, Tensor, float, Tensor, bool]) -> None:
        """
        Stores a transition in memory with a priority based on the absolute error.

        Args:
            transition (Tuple[Tensor, Tensor, float, Tensor, bool]):
                - state (Tensor): Current state observation.
                - action (Tensor): Action taken.
                - reward (float): Reward received.
                - next_state (Tensor): Next state observation.
                - done (bool): Whether the episode has terminated.
        """
        idx = self.sum_tree.add(0, transition)

        # Priority is the absolute error of the TD update
        td_error = transition[-1]
        priority = np.abs(td_error) + 1e-6  # Avoid zero priority

        self.sum_tree.update(idx, priority)

    def sample(self, batch_size: int) -> Dict[str, Tensor]:
        """
        Samples a batch of transitions based on their priorities.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Dict[str, Tensor]: Batched transitions with keys:
                - states (Tensor): Batched current states.
                - actions (Tensor): Batched actions.
                - rewards (Tensor): Batched rewards.
                - next_states (Tensor): Batched next states.
                - dones (Tensor): Batched episode termination flags.
                - weights (Tensor): Importance sampling weights.
                - batch_indices (Tensor): Original indices of sampled transitions.
        """
        if len(self.sum_tree.tree) < MIN_OBSERVATION_SIZE:
            raise MemoryError(
                "Memory does not have enough observations for sampling."
            )

        batch_indices = []
        priorities = []
        priority_segment = self.sum_tree.tree[0]
        p_total = priority_segment
        segment = 0

        for i in range(batch_size):
            a = p_total * np.random.rand()
            b = priority_segment
            idx, priority = self.sum_tree._retrieve(segment, a)
            priorities.append(priority)
            batch_indices.append(idx)

            p_total -= b
            segment = idx if idx < self.sum_tree.capacity else idx - self.sum_tree.capacity

        states = Tensor([self.sum_tree.data[i][0] for i in batch_indices])
        actions = Tensor([self.sum_tree.data[i][1] for i in batch_indices])
        rewards = Tensor([self.sum_وتوزع_tree.data[i][2] for i in batch_indices])
        next_states = Tensor([self.sum_tree.data[i][3] for i in batch_indices])
        dones = Tensor([self.sum_tree.data[i][4] for i in batch_indices])
        weights = Tensor(
            priorities ** (-self.beta) / max(priorities ** (-self.beta))
        )
        weights /= weights.sum()

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "weights": weights,
            "batch_indices": Tensor(batch_indices),
        }

    def update_priorities(self, batch_indices: List[int], td_errors: List[float]) -> None:
        """
        Updates the priorities of sampled transitions based on their TD errors.

        Args:
            batch_indices (List[int]): Original indices of sampled transitions.
            td_errors (List[float]): Temporal difference errors for the sampled transitions.
        """
        for idx, error in zip(batch_indices, td_errors):
            error = np.abs(error) + 1e-6  # Avoid zero priority
            self.sum_tree.update(idx, error)

        self.beta = min(1., self.beta + self.beta_increment)

    def __len__(self) -> int:
        """
        Returns the current number of stored transitions.

        Returns:
            int: Length of the memory.
        """
        return len(self.sum_tree.data)

# Main class for experience replay and memory management
class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    """

    def __init__(
        self,
        memory_size: int = MEMORY_SIZE,
        prioritized: bool = False,
        velocity_threshold: float = VELOCITY_THRESHOLD,
    ):
        """
        Initializes the replay buffer.

        Args:
            memory_size (int): Maximum number of transitions to store.
            prioritized (bool): Whether to use prioritized experience replay.
            velocity_threshold (float): Threshold for velocity-based prioritization.
        """
        self.memory_size = memory_size
        self.prioritized = prioritized
        self.velocity_threshold = velocity_threshold

        if prioritized:
            self.memory = PrioritizedReplayMemory(memory_size)
        else:
            self.memory = ExperienceReplayMemory(memory_size)

    def store_transition(
        self,
        state: Tensor,
        action: Tensor,
        reward: float,
        next_state: Tensor,
        done: bool,
        velocity: Optional[float] = None,
    ) -> None:
        """
        Stores a transition in the replay buffer.

        Args:
            state (Tensor): Current state observation.
            action (Tensor): Action taken.
            reward (float): Reward received.
            next_state (Tensor): Next state observation.
            done (bool): Whether the episode has terminated.
            velocity (Optional[float]): Velocity of the agent (for velocity-based prioritization).
        """
        td_error = reward if done else reward + self.memory.memory[-1][-1]
        transition = (state, action, reward, next_state, done, td_error)

        if self.prioritized:
            if velocity is not None:
                # Velocity-based prioritization
                if np.abs(velocity) > self.velocity_threshold:
                    priority = 1
                else:
                    priority = 0
            else:
                priority = 0

            self.memory.remember(transition)
        else:
            self.memory.remember(transition)

    def sample_batch(self, batch_size: int) -> Dict[str, Tensor]:
        """
        Samples a batch of transitions for training.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Dict[str, Tensor]: Batched transitions with keys:
                - states (Tensor): Batched current states.
                - actions (Tensor): Batched actions.
                - rewards (Tensor): Batched rewards.
                - next_states (Tensor): Batched next states.
                - dones (Tensor): Batched episode termination flags.
                - weights (Tensor, optional): Importance sampling weights (for prioritized replay).
                - batch_indices (Tensor, optional): Original indices of sampled transitions.
        """
        batch = self.memory.sample(batch_size)

        if self.prioritized:
            return {
                "states": batch["states"],
                "actions": batch["actions"],
                "rewards": batch["rewards"],
                "next_states": batch["next_states"],
                "dones": batch["dones"],
                "weights": batch["weights"],
                "batch_indices": batch["batch_indices"],
            }
        else:
            return {
                "states": batch["states"],
                "actions": batch["actions"],
                "rewards": batch["rewards"],
                "next_states": batch["next_states"],
                "dones": batch["dones"],
            }

    def update_priorities(self, batch_indices: List[int], td_errors: List[float]) -> None:
        """
        Updates the priorities of sampled transitions based on their TD errors.

        Args:
            batch_indices (List[int]): Original indices of sampled transitions.
            td_errors (List[float]): Temporal difference errors for the sampled transitions.
        """
        if self.prioritized:
            self.memory.update_priorities(batch_indices, td_errors)

    def __len__(self) -> int:
        """
        Returns the current number of stored transitions.

        Returns:
            int: Length of the replay buffer.
        """
        return len(self.memory)

# Helper function for updating priorities based on velocity
def update_priorities_with_velocity(
    replay_buffer: ReplayBuffer,
    velocities: List[float],
    batch_indices: List[int],
) -> None:
    """
    Updates the priorities of sampled transitions based on the agent's velocities.

    Args:
        replay_buffer (ReplayBuffer): Replay buffer to update.
        velocities (List[float]): Velocities of the agent for each transition.
        batch_indices (List[int]): Original indices of sampled transitions.
    """
    for idx, velocity in zip(batch_indices, velocities):
        priority = 1 if np.abs(velocity) > replay_buffer.velocity_threshold else 0
        replay_buffer.memory.update(idx, priority)

# Helper function for logging memory statistics
def log_memory_statistics(replay_buffer: ReplayBuffer) -> pd.DataFrame:
    """
    Logs memory statistics and returns them as a DataFrame.

    Args:
        replay_buffer (ReplayBuffer): Replay buffer to log statistics for.

    Returns:
        pd.DataFrame: Memory statistics including memory size, fill rate, and priority distribution.
    """
    memory = replay_buffer.memory
    data = {
        "memory_size": memory.memory_size,
        "length": len(memory),
        "fill_rate": len(memory) / memory.memory_size,
    }

    if isinstance(memory, PrioritizedReplayMemory):
        priorities = [transition[1] for transition in memory.sum_tree.data]
        data["priority_mean"] = np.mean(priorities)
        data["priority_std"] = np.std(priorities)
        data["priority_min"] = np.min(priorities)
        data["priority_max"] = np.max(priorities)

    df = pd.DataFrame(data, index=[0])
    logger.info("Memory statistics:\n%s", df)
    return df

# Main experience replay and memory interface
class ExperienceReplay:
    """
    Main interface for experience replay and memory management.
    """

    def __init__(
        self,
        memory_size: int = MEMORY_SIZE,
        prioritized: bool = False,
        velocity_threshold: float = VELOCITY_THRESHOLD,
    ):
        """
        Initializes the experience replay and memory module.

        Args:
            memory_size (int): Maximum number of transitions to store.
            prioritized (bool): Whether to use prioritized experience replay.
            velocity_threshold (float): Threshold for velocity-based prioritization.
        """
        self.memory_size = memory_size
        self.prioritized = prioritized
        self.velocity_threshold = velocity_threshold
        self.replay_buffer = ReplayBuffer(
            memory_size, prioritized, velocity_threshold
        )

    def store_transition(
        self,
        state
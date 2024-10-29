import pickle
import numpy as np
import dataclasses
from datetime import datetime
from pathlib import Path

from dataclasses import dataclass
from datasets import Dataset
from typing import TypeVar, Generic, List, Iterator, Optional, Type, Dict


@dataclass
class TrainConfigR:
    task: str
    data_dir: str
    batch_size: int
    shuffle: bool
    in_features: int
    out_features: int
    lr: float
    lr_decay: float
    decay_every: int
    epochs: int
    results_path: Optional[str]
    seed: int


@dataclass
class TrainConfigC:
    task: str
    data_dir: str
    mean: float
    in_features: int
    lr: float
    lr_decay: float
    decay_every: int
    steps: int
    results_path: Optional[str]
    seed: int


class Parameter(np.ndarray):
    r"""A parameter class for storing model parameters

    This class is a subclass of numpy.ndarray and is used to store model
    parameters. It is created by calling `Parameter` on a numpy array.

    Example:
        >>> import numpy as np
        >>> from utils import Parameter
        >>> param = Parameter(np.array([1, 2, 3]))
        >>> print(param)
        [1 2 3]
    """

    def __new__(cls, input_array):
        # Create a new instance of Parameter
        obj = np.asarray(input_array).view(cls)
        return obj


class Loss:
    r"""Base class for all loss functions

    All other loss functions should subclass this class and implement the
    `__call__` and `backward` methods.
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        r"""Compute the loss

        Args:
            y_pred: The predicted values
            y_true: The true values
        """
        raise NotImplementedError

    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
        r"""Compute the gradients of the loss with respect to the parameters

        Args:
            x: The input values
            y_pred: The predicted values
            y_true: The true values

        Returns:
            The gradients of the loss with respect to the parameters
        """
        raise NotImplementedError


class SGD:
    r"""Stochastic gradient descent optimizer.

    This optimizer updates the parameters using stochastic gradient descent.

    Attributes:
        params: The parameters to optimize
        lr: The learning rate
        lr_lower_bound: The lower bound of the learning rate
        opt_step: The current optimization step
        lr_decay: The learning rate decay factor
        decay_every: The number of steps after which to decay the learning rate

    Methods:
        step: Update the parameters with the gradients
    """

    def __init__(self, params: Iterator, lr: float, lr_decay: float = 0.99, decay_every: int = 10):
        r"""Initialize the optimizer.

        Args:
            params: The parameters to optimize
            lr: The learning rate
            lr_decay: The learning rate decay factor
            decay_every: The number of steps after which to decay the learning rate
        """
        self.params = list(params)
        self.lr = lr
        self.lr_lower_bound = lr / 100
        self.opt_step = 0
        self.lr_decay = lr_decay
        self.decay_every = decay_every

    def step(self, grads: dict[str, np.ndarray]):
        r"""Update the parameters with the gradients

        Args:
            grads: The gradients of the parameters
        """
        for name, param in self.params:
            param -= self.lr * grads[name]
        self.opt_step += 1
        if self.opt_step % self.decay_every == 0:
            self.lr = max(self.lr_lower_bound, self.lr * self.lr_decay)


class GD:
    r"""Gradient descent optimizer.

    This optimizer updates the parameters using gradient descent.

    Attributes:
        params: The parameters to optimize
        lr: The learning rate
        lr_lower_bound: The lower bound of the learning rate
        opt_step: The current optimization step
        lr_decay: The learning rate decay factor
        decay_every: The number of steps after which to decay the learning rate

    Methods:
        step: Update the parameters with the gradients
    """

    def __init__(self, params, lr, lr_decay=0.99, decay_every=10):
        self.params = list(params)
        self.lr = lr
        self.lr_lower_bound = lr / 100
        self.opt_step = 0
        self.lr_decay = lr_decay
        self.decay_every = decay_every

    def step(self, grads: dict[str, np.ndarray]):
        r"""Update the parameters with the gradients

        Args:
            grads: The gradients of the parameters

        Returns:
            True if the optimization has converged
        """
        for name, param in self.params:
            param -= self.lr * grads[name]
        self.opt_step += 1
        if self.opt_step % self.decay_every == 0:
            self.lr = max(self.lr_lower_bound, self.lr * self.lr_decay)


T = TypeVar("T")


class DataLoader(Generic[T]):
    r"""A simple data loader for iterating over a dataset

    This data loader takes a dataset and returns batches of data.

    Attributes:
        dataset: The dataset to iterate over
        batch_size: The batch size
        shuffle: Whether to shuffle the data
        train: Whether the data loader is used for training
        index: The current index in the dataset

    Methods:
        _reset_indices: Reset the indices of the dataset
        __iter__: Return the iterator object
        __next__: Return the next batch of data
        __len__: Return the number of batches in the dataset
        _remove_index_column: Remove the index column from the data
    """

    def __init__(self, dataset: Type[T], batch_size: int, shuffle: bool = True, train: bool = False):
        r"""Initialize the data loader.

        Args:
            dataset: The dataset to iterate over
            batch_size: The batch size
            shuffle: Whether to shuffle the data
            train: Whether the data loader is used for training
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.index = 0
        self._reset_indices()

    def _reset_indices(self):
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))
        else:
            self.indices = np.arange(len(self.dataset))
        self.index = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        self._reset_indices()
        return self

    def __next__(self) -> np.ndarray:
        if self.index >= len(self.dataset):
            if self.train:
                self._reset_indices()  # 重新初始化以便下一次迭代
            else:
                raise StopIteration
        batch = np.array(
            [
                self._remove_index_column(self.dataset[int(i)])
                for i in self.indices[self.index : self.index + self.batch_size]
            ]
        )
        self.index += self.batch_size
        return batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _remove_index_column(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        if "__index_level_0__" in data:
            del data["__index_level_0__"]
        return np.array(list(data.values()))


def save(state_dict: dict[str, np.ndarray], path: str):
    r"""Save the state_dict as pkl

    Args:
        state_dict: The state_dict of a model
        path: Where the state will be stored
    """
    with open(path, "wb") as f:
        pickle.dump(state_dict, f)


def load(path: str) -> dict[str, np.ndarray]:
    r"""Load state_dict from disk

    Args:
        path: Where the state will be stored
    """
    with open(path, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def init_config_from_args(cls, args):
    """Initialize a dataclass from a Namespace."""
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def get_date_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def handle_results_path(res_path: str, default_root: str = "./results") -> Path:
    """Sets results path if it doesn't exist yet."""
    if res_path is None:
        results_path = Path(default_root) / get_date_str()
    else:
        results_path = Path(res_path)
    print(f"Results will be saved to '{results_path}'")
    return results_path

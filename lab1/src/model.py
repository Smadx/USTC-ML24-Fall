import numpy as np
from typing import Dict, Optional, Any, Callable, Iterator, Tuple
from collections import OrderedDict
from utils import Parameter


def _predict_unimplemented(self, *input: Any) -> None:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`BaseModel` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(f'Model [{type(self).__name__}] is missing the required "predict" function')


class BaseModel:
    r"""Base class for all your models.

    Your models should also subclass this class.

    Example::

        >>> from model import BaseModel
        >>> from utils import save
        >>> # Define your model
        >>> class YourModel(BaseModel):
        >>>     def __init__(self, *args, **kwargs):
        >>>         super.__init__()
        >>>         self.param = Parameter(...)
        >>>
        >>>     def predict(self, input: np.ndarry):
        >>>         # Details
        >>>
        >>> # Use the model to predict
        >>> model = Yourmodel()
        >>> pred = model(inputs)
        >>>
        >>> # Save the parameters
        >>> state_dict = model.state_dict()
        >>> save(state_dict, 'model.pkl')
    """

    training: bool = False
    _parameters: Dict[str, Optional[Parameter]]
    call_super_init: bool = True

    def __init__(self, *args, **kwargs) -> None:
        if self.call_super_init is False and bool(kwargs):
            raise TypeError(
                "{}.__init__() got an unexpected keyword argument '{}'"
                "".format(type(self).__name__, next(iter(kwargs)))
            )

        if self.call_super_init is False and bool(args):
            raise TypeError(
                f"{type(self).__name__}.__init__() takes 1 positional argument but {len(args) + 1} were" " given"
            )

        """
        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Model.__setattr__ overhead. Model's __setattr__ has special
        handling for parameters, subModels, and buffers but simply calls into
        super().__setattr__ for all other attributes.
        """
        super().__setattr__("training", True)
        super().__setattr__("_parameters", OrderedDict())

        if self.call_super_init:
            super().__init__(*args, **kwargs)

    predict: Callable[..., Any] = _predict_unimplemented

    def _call_impl(self, *args, **kwargs):
        predict_call = self.predict
        return predict_call(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def _save_state_dict(self, destination: dict):
        r"""Saves the state of the model to a dictionary.

        Args:
            destination (dict): A dict where state will be stored
        """
        for name, param in self._parameters.items():
            destination[name] = param

    def state_dict(self, destination: Optional[dict] = None):
        r"""Returns the state dict of model

        Args:
            destination (dict): A dict where state will be stored
        """
        if destination is None:
            destination = OrderedDict()
        self._save_state_dict(destination)
        return destination

    def load_from_state_dict(self, state_dict: dict):
        r"""Loads the state of the model from a dictionary.

        Args:
            state_dict (dict): A dictionary containing the state of the model.
        """
        for name, param in state_dict.items():
            self._parameters[name] = param
            setattr(self, name, param)

    def parameters(self) -> Iterator[Tuple[str, Parameter]]:
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Example::

            >>> from model import LinearRegression
            >>> from utils import SGD
            >>> model = LinearRegression(3, 1)
            >>> optimizer = SGD(model.parameters(), lr=0.01)
        """
        for name, param in self._parameters.items():
            yield name, param

    def train(self) -> None:
        r"""Sets the Model in training mode."""
        self.training = True

    def eval(self) -> None:
        r"""Sets the Model in evaluation mode."""
        self.training = False

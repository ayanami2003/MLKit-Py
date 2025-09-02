from typing import Any, Callable, Optional
from general.structures.data_batch import DataBatch
import threading

class DeferredEvaluator:
    """
    A utility class that encapsulates deferred (lazy) evaluation of operations on data.

    This class enables performance optimization by delaying the execution of transformations
    or computations until the result is explicitly requested. It supports chaining of
    operations to build up a computation graph which is executed only when needed.

    Attributes:
        _data (Any): The underlying data or a reference to it.
        _evaluator (Optional[Callable]): The function to be called for evaluation.
        _args (tuple): Positional arguments for the evaluator.
        _kwargs (dict): Keyword arguments for the evaluator.
        _is_evaluated (bool): Flag indicating if the evaluation has been performed.
        _result (Any): Cached result after evaluation.
    """

    def __init__(self, data: Any, evaluator: Optional[Callable]=None, *args: Any, **kwargs: Any):
        """
        Initialize a DeferredEvaluator instance.

        Args:
            data (Any): The initial data or a reference to be processed.
            evaluator (Optional[Callable]): The function that performs the actual evaluation.
            *args (Any): Positional arguments to pass to the evaluator.
            **kwargs (Any): Keyword arguments to pass to the evaluator.
        """
        self._data = data
        self._evaluator = evaluator
        self._args = args
        self._kwargs = kwargs
        self._is_evaluated = False
        self._result = None
        self._lock = threading.Lock()

    def evaluate(self) -> Any:
        """
        Trigger the deferred evaluation if not already performed and return the result.

        Returns:
            Any: The result of the evaluation.
        """
        if not self._is_evaluated:
            with self._lock:
                if not self._is_evaluated:
                    if self._evaluator is None:
                        self._result = self._data
                    else:
                        self._result = self._evaluator(self._data, *self._args, **self._kwargs)
                    self._is_evaluated = True
        return self._result

    def is_ready(self) -> bool:
        """
        Check if the evaluator is ready to perform evaluation.

        Returns:
            bool: True if evaluator is set, False otherwise.
        """
        return self._evaluator is not None

    def chain_operation(self, next_evaluator: Callable, *args: Any, **kwargs: Any) -> 'DeferredEvaluator':
        """
        Chain another operation to be performed after the current one.

        Args:
            next_evaluator (Callable): The next function to evaluate.
            *args (Any): Positional arguments for the next evaluator.
            **kwargs (Any): Keyword arguments for the next evaluator.

        Returns:
            DeferredEvaluator: A new instance representing the chained operation.
        """

        def composed_evaluator(data: Any) -> Any:
            if self.is_ready():
                intermediate_result = self.evaluate()
            else:
                intermediate_result = data
            return next_evaluator(intermediate_result, *args, **kwargs)
        if self.is_ready():
            return DeferredEvaluator(None, composed_evaluator)
        else:
            return DeferredEvaluator(self._data, composed_evaluator)
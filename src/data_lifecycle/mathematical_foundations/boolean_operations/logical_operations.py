from typing import Union, List, Callable

def and_operation(a: Union[bool, List[bool]], b: Union[bool, List[bool]]) -> Union[bool, List[bool]]:
    """
    Perform AND operation on boolean values or lists of boolean values.

    This function computes the logical AND operation between two boolean inputs.
    If both inputs are single boolean values, it returns a single boolean result.
    If both inputs are lists of booleans, it performs element-wise AND and returns a list
    of boolean results. The function expects both lists to be of equal length.

    Args:
        a (Union[bool, List[bool]]): First boolean operand or list of boolean operands.
        b (Union[bool, List[bool]]): Second boolean operand or list of boolean operands.

    Returns:
        Union[bool, List[bool]]: Result of AND operation as a boolean or list of booleans.

    Raises:
        ValueError: If input lists are of unequal length.
        TypeError: If inputs are neither boolean nor list of booleans.
    """
    if isinstance(a, bool) and isinstance(b, bool):
        return a and b
    if isinstance(a, list) and isinstance(b, list):
        if not all((isinstance(x, bool) for x in a)):
            raise TypeError('All elements in the first list must be boolean values.')
        if not all((isinstance(x, bool) for x in b)):
            raise TypeError('All elements in the second list must be boolean values.')
        if len(a) != len(b):
            raise ValueError('Input lists must have equal length.')
        return [x and y for (x, y) in zip(a, b)]
    raise TypeError('Inputs must be either boolean values or lists of boolean values.')

def xor_operation(a: Union[bool, List[bool]], b: Union[bool, List[bool]]) -> Union[bool, List[bool]]:
    """
    Perform XOR operation on boolean values or lists of boolean values.

    This function computes the exclusive OR (XOR) operation between two boolean inputs.
    If both inputs are single boolean values, it returns a single boolean result.
    If both inputs are lists of booleans, it performs element-wise XOR and returns a list
    of boolean results. The function expects both lists to be of equal length.

    Args:
        a (Union[bool, List[bool]]): First boolean operand or list of boolean operands.
        b (Union[bool, List[bool]]): Second boolean operand or list of boolean operands.

    Returns:
        Union[bool, List[bool]]: Result of XOR operation as a boolean or list of booleans.

    Raises:
        ValueError: If input lists are of unequal length.
        TypeError: If inputs are neither boolean nor list of booleans.
    """
    if isinstance(a, bool) and isinstance(b, bool):
        return a ^ b
    if isinstance(a, list) and isinstance(b, list):
        if not all((isinstance(x, bool) for x in a)):
            raise TypeError('All elements in the first list must be boolean values.')
        if not all((isinstance(x, bool) for x in b)):
            raise TypeError('All elements in the second list must be boolean values.')
        if len(a) != len(b):
            raise ValueError('Input lists must have equal length.')
        return [x ^ y for (x, y) in zip(a, b)]
    raise TypeError('Inputs must be either boolean values or lists of boolean values.')

def short_circuit_and_operation(a: Union[bool, Callable[[], bool]], b: Union[bool, Callable[[], bool]]) -> bool:
    """
    Perform short-circuit AND operation on boolean values or lazy-evaluated conditions.

    This function implements short-circuit evaluation for the AND operation. If the first operand
    evaluates to False, the second operand is never evaluated (saving computation). Operands can
    be direct boolean values or callable functions that return boolean values (for lazy evaluation).

    Args:
        a (Union[bool, Callable[[], bool]]): First operand - either a boolean or a callable returning a boolean.
        b (Union[bool, Callable[[], bool]]): Second operand - either a boolean or a callable returning a boolean.

    Returns:
        bool: Result of short-circuit AND operation.

    Raises:
        TypeError: If operands are neither boolean nor callable, or if callable doesn't return a boolean.
    """
    if isinstance(a, bool):
        a_val = a
    elif callable(a):
        a_val = a()
        if not isinstance(a_val, bool):
            raise TypeError("Callable operand 'a' must return a boolean value.")
    else:
        raise TypeError("Operand 'a' must be either a boolean or a callable returning a boolean.")
    if not a_val:
        return False
    if isinstance(b, bool):
        b_val = b
    elif callable(b):
        b_val = b()
        if not isinstance(b_val, bool):
            raise TypeError("Callable operand 'b' must return a boolean value.")
    else:
        raise TypeError("Operand 'b' must be either a boolean or a callable returning a boolean.")
    return a_val and b_val
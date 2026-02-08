"""Calibration of differentially private mechanisms.

Searches for optimal mechanism parameter values that achieve a target
(epsilon, delta)-DP guarantee.

API-compatible with ``dp_accounting.mechanism_calibration``.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from dp_accelerator.dp_event import DpEvent, NoOpDpEvent
from dp_accelerator.privacy_accountant import PrivacyAccountant


class BracketInterval:
    """Base class for bracket intervals."""
    pass


class ExplicitBracketInterval(BracketInterval):
    """Explicit bracket interval with two endpoints.

    Attributes:
        endpoint_1: First endpoint.
        endpoint_2: Second endpoint.
    """

    def __init__(self, endpoint_1: float, endpoint_2: float):
        self.endpoint_1 = endpoint_1
        self.endpoint_2 = endpoint_2

    def __repr__(self):
        return (
            f"ExplicitBracketInterval("
            f"endpoint_1={self.endpoint_1}, endpoint_2={self.endpoint_2})"
        )


class LowerEndpointAndGuess(BracketInterval):
    """Lower endpoint and an initial guess for exponential search.

    Attributes:
        lower_endpoint: Lower bound for the search.
        initial_guess: Starting point for exponential expansion.
    """

    def __init__(self, lower_endpoint: float, initial_guess: float):
        self.lower_endpoint = lower_endpoint
        self.initial_guess = initial_guess


class NoBracketIntervalFoundError(Exception):
    """Raised when no valid bracket interval can be found."""


class NonEmptyAccountantError(Exception):
    """Raised when make_fresh_accountant returns an accountant with non-empty ledger."""


def _search_for_explicit_bracket_interval(
    bracket_interval: LowerEndpointAndGuess,
    epsilon_gap: Callable[[float], float],
) -> ExplicitBracketInterval:
    """Expand a LowerEndpointAndGuess into an ExplicitBracketInterval."""
    lower = bracket_interval.lower_endpoint
    upper = bracket_interval.initial_guess
    if lower >= upper:
        raise ValueError(
            f"bracket_interval.lower_endpoint ({lower}) must be less than "
            f"bracket_interval.initial_guess ({upper})."
        )

    lower_value = epsilon_gap(lower)
    upper_value = epsilon_gap(upper)
    gap = upper - lower
    num_tries = 0

    while lower_value * upper_value > 0:
        num_tries += 1
        if num_tries > 30:
            raise NoBracketIntervalFoundError(
                "Unable to find bracketing interval within 2**30 of initial "
                "guess. Consider providing an ExplicitBracketInterval."
            )
        gap *= 2
        lower, upper = upper, upper + gap
        lower_value, upper_value = upper_value, epsilon_gap(upper)

    return ExplicitBracketInterval(lower, upper)


def _bisect(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    tol: float,
    lower_value: Optional[float] = None,
    upper_value: Optional[float] = None,
) -> float:
    """Bisection search for approximate root with non-positive value."""
    if lower_value is None:
        lower_value = function(lower)
    if upper_value is None:
        upper_value = function(upper)

    if lower_value == 0:
        return lower
    if upper_value == 0:
        return upper
    if lower_value * upper_value > 0:
        raise ValueError("Values must have opposite signs.")
    if upper - lower <= tol:
        return lower if lower_value < 0 else upper

    middle = (lower + upper) / 2.0
    middle_value = function(middle)

    if middle_value == 0:
        return middle
    elif lower_value * middle_value < 0:
        return _bisect(function, lower, middle, tol, lower_value, middle_value)
    else:
        return _bisect(function, middle, upper, tol, middle_value, upper_value)


def calibrate_dp_mechanism(
    make_fresh_accountant: Callable[[], PrivacyAccountant],
    make_event_from_param: Union[
        Callable[[float], DpEvent], Callable[[int], DpEvent]
    ],
    target_epsilon: float,
    target_delta: float,
    bracket_interval: Optional[BracketInterval] = None,
    discrete: bool = False,
    tol: Optional[float] = None,
) -> Union[float, int]:
    """Search for optimal mechanism parameter within privacy budget.

    Uses Brent's method (via scipy) or fallback bisection to find the
    parameter value at which the target epsilon is achieved.

    Args:
        make_fresh_accountant: Callable returning a freshly initialized
            PrivacyAccountant.
        make_event_from_param: Callable mapping parameter value to a DpEvent.
        target_epsilon: Target epsilon (>= 0).
        target_delta: Target delta (in [0, 1]).
        bracket_interval: Search bracket. If None, uses [0, 1].
        discrete: If True, parameter is integer-valued.
        tol: Search tolerance. Defaults to 1e-6 (continuous) or 1.0 (discrete).

    Returns:
        The optimal parameter value (float or int).
    """
    if not callable(make_fresh_accountant):
        raise TypeError(
            f"make_fresh_accountant must be callable. Found {type(make_fresh_accountant)}."
        )
    if not callable(make_event_from_param):
        raise TypeError(
            f"make_event_from_param must be callable. Found {type(make_event_from_param)}."
        )
    if target_epsilon < 0:
        raise ValueError(f"target_epsilon must be nonnegative. Found {target_epsilon}.")
    if not 0 <= target_delta <= 1:
        raise ValueError(f"target_delta must be in [0, 1]. Found {target_delta}.")

    if bracket_interval is None:
        bracket_interval = LowerEndpointAndGuess(0, 1)

    if tol is None:
        tol = 1.0 if discrete else 1e-6
    elif discrete:
        tol = max(tol, 1.0)
    elif tol <= 0:
        raise ValueError(f"tol must be positive. Found {tol}.")

    def epsilon_gap(x: float) -> float:
        if discrete:
            x = round(x)
        event = make_event_from_param(x)
        accountant = make_fresh_accountant()
        if not isinstance(accountant.ledger, NoOpDpEvent):
            raise NonEmptyAccountantError()
        return accountant.compose(event).get_epsilon(target_delta) - target_epsilon

    if isinstance(bracket_interval, LowerEndpointAndGuess):
        bracket_interval = _search_for_explicit_bracket_interval(
            bracket_interval, epsilon_gap
        )
    elif not isinstance(bracket_interval, ExplicitBracketInterval):
        raise TypeError(
            f"Unrecognized bracket_interval type: {type(bracket_interval)}"
        )

    # Try scipy.optimize.brentq if available, fallback to bisection
    try:
        from scipy import optimize

        try:
            root, result = optimize.brentq(
                epsilon_gap,
                bracket_interval.endpoint_1,
                bracket_interval.endpoint_2,
                xtol=tol,
                full_output=True,
            )
        except ValueError as err:
            raise ValueError(
                f"`brentq` raised ValueError. The bracket interval "
                f"{bracket_interval} may not bracket a solution."
            ) from err

        if not result.converged:
            root = None
        else:
            if epsilon_gap(root) > 0:
                if epsilon_gap(root + tol) < 0:
                    root += tol
                elif epsilon_gap(root - tol) < 0:
                    root -= tol
                else:
                    root = None

        if root is None:
            root = _bisect(
                epsilon_gap,
                bracket_interval.endpoint_1,
                bracket_interval.endpoint_2,
                tol,
            )
    except ImportError:
        # No scipy — use bisection directly
        root = _bisect(
            epsilon_gap,
            bracket_interval.endpoint_1,
            bracket_interval.endpoint_2,
            tol,
        )

    if discrete:
        root = round(root)

    return root

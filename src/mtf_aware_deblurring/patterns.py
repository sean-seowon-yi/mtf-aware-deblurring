from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

import numpy as np


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    limit = int(n**0.5) + 1
    for factor in range(3, limit, 2):
        if n % factor == 0:
            return False
    return True


def next_prime(n: int) -> int:
    candidate = max(2, int(n))
    if candidate <= 2:
        return 2
    if candidate % 2 == 0:
        candidate += 1
    while not is_prime(candidate):
        candidate += 2
    return candidate


def previous_prime(n: int) -> Optional[int]:
    candidate = int(n)
    if candidate < 2:
        return None
    if candidate == 2:
        return 2
    if candidate % 2 == 0:
        candidate -= 1
    while candidate >= 3:
        if is_prime(candidate):
            return candidate
        candidate -= 2
    return 2 if n >= 2 else None


def legendre_symbol(a: int, p: int) -> int:
    if p <= 2 or not is_prime(p):
        raise ValueError("Legendre symbol is defined for odd prime p >= 3.")
    a %= p
    if a == 0:
        return 0
    res = pow(a, (p - 1) // 2, p)
    if res == p - 1:
        return -1
    return res


def resolve_legendre_prime(length: int, prime: Optional[int]) -> int:
    if prime is not None:
        if not is_prime(prime):
            raise ValueError(f"Provided prime {prime} is not prime.")
        if prime < 3 or prime % 2 == 0:
            raise ValueError("Legendre sequence requires an odd prime >= 3.")
        return prime

    if length >= 3:
        candidate = previous_prime(length)
        if candidate is None or candidate < 3:
            candidate = next_prime(length)
    else:
        candidate = 3

    if candidate % 2 == 0:
        candidate = next_prime(candidate + 1)
    return candidate


def legendre_base_sequence(prime: int) -> np.ndarray:
    if not is_prime(prime) or prime < 3 or prime % 2 == 0:
        raise ValueError("Legendre sequence requires an odd prime length >= 3.")
    seq = np.zeros(prime, dtype=np.float64)
    seq[0] = 1.0
    for i in range(1, prime):
        seq[i] = 1.0 if legendre_symbol(i, prime) == 1 else 0.0
    return seq


def modified_legendre_sequence(
    length: int,
    prime: Optional[int] = None,
    rotation: int = 0,
    append_mode: Literal["auto", "append", "truncate", "repeat"] = "auto",
    flip: bool = False,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("Sequence length must be positive.")

    base_prime = resolve_legendre_prime(length, prime)
    base = legendre_base_sequence(base_prime)
    rotation = int(rotation) % base_prime
    rotated = np.roll(base, -rotation)

    if append_mode == "auto":
        if length <= base_prime:
            mode = "truncate"
        elif length <= 2 * base_prime:
            mode = "append"
        else:
            mode = "repeat"
    else:
        mode = append_mode

    if mode == "truncate":
        if length > base_prime:
            raise ValueError("truncate mode requires length <= prime.")
        code = rotated[:length].copy()
    elif mode == "append":
        if length <= base_prime:
            code = rotated[:length].copy()
        else:
            if length > 2 * base_prime:
                raise ValueError("append mode supports lengths up to 2 * prime.")
            extra = length - base_prime
            extra = int(extra)
            code = np.concatenate([rotated, rotated[:extra]])
    elif mode == "repeat":
        reps, remainder = divmod(length, base_prime)
        segments = []
        if reps > 0:
            segments.append(np.tile(rotated, reps))
        if remainder:
            segments.append(rotated[:remainder])
        if segments:
            code = np.concatenate(segments)
        else:
            code = rotated[:length].copy()
    else:
        raise ValueError(f"Unknown append_mode '{mode}'.")

    if flip:
        code = 1.0 - code

    code = code.astype(np.float64, copy=False)
    if code.size != length:
        raise RuntimeError("Generated Legendre code has incorrect length.")
    if code.sum() == 0:
        code[0] = 1.0
    return code


def make_exposure_code(
    T: int,
    pattern: Literal["box", "random", "legendre"] = "box",
    seed: Optional[int] = 0,
    duty_cycle: float = 0.5,
    legendre_params: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    if pattern == "box":
        return np.ones(T, dtype=float)
    if pattern == "random":
        rng = np.random.default_rng(seed)
        code = (rng.random(T) < duty_cycle).astype(float)
        if code.sum() == 0:
            code[rng.integers(0, T)] = 1.0
        return code
    if pattern == "legendre":
        params = dict(legendre_params or {})
        if "prime" not in params:
            params["prime"] = resolve_legendre_prime(T, None)
        if "rotation" not in params and seed is not None:
            params["rotation"] = int(seed) % max(T, 1)
        code = modified_legendre_sequence(length=T, **params)
        if code.sum() == 0:
            code[0] = 1.0
        return code
    raise ValueError("Unknown pattern")


__all__ = [
    "is_prime",
    "next_prime",
    "previous_prime",
    "legendre_symbol",
    "resolve_legendre_prime",
    "legendre_base_sequence",
    "modified_legendre_sequence",
    "make_exposure_code",
]

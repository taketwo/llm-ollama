"""Caching for expensive function calls."""

import functools
import inspect
from pathlib import Path
from typing import Any, Dict

import yaml


class Cache:
    """A generic caching mechanism for expensive function calls.

    This class provides a decorator that can be used to cache the results of function
    calls based on specific parameter values. The cache is stored in YAML files in a
    specified directory.

    Attributes
    ----------
    CACHE_VERSION : int
        The current cache version. This will be incremented if the cache format changes
        or there is a need to invalidate existing caches for any reason.
    cache_dir : Path
        The directory where cache files are stored.

    """

    CACHE_VERSION = 2

    def __init__(self, cache_dir: Path) -> None:
        """Initialize a cache with the specified directory.

        Parameters
        ----------
        cache_dir : Path
            Path to the directory where cache files will be stored. The directory will
            be created if it does not exist.

        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, cache_name: str, key: str):
        """Decorate function to cache results based on parameter values.

        Parameters
        ----------
        cache_name : str
            The name of the cache file (without extension). This will be used to create
            a YAML file in the cache directory.
        key : str
            Parameter name to use as the cache key. This must be a string that matches
            one of the parameter names in the decorated function.

        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string parameter name")

        cache_file = self.cache_dir / f"{cache_name}.yaml"

        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if key in kwargs:
                    cache_key = kwargs[key]
                elif key in param_names and param_names.index(key) < len(args):
                    cache_key = args[param_names.index(key)]
                else:
                    raise ValueError(f"Parameter '{key}' not provided to function")

                # Make sure cache_key is serializable for YAML
                try:
                    cache_key = str(cache_key)
                except TypeError as e:
                    raise ValueError(
                        f"Parameter '{key}' is not serializable for YAML",
                    ) from e

                cache: Dict[str, Any] = {}
                try:
                    with cache_file.open("r") as f:
                        loaded_cache = yaml.safe_load(f)
                        if loaded_cache is not None:
                            cache = loaded_cache
                except (FileNotFoundError, yaml.scanner.ScannerError):
                    pass

                # Invalidate cache if version is not present or has changed
                if "version" not in cache or cache["version"] != self.CACHE_VERSION:
                    cache = {"version": self.CACHE_VERSION, "data": {}}

                if cache_key in cache["data"]:
                    return cache["data"][cache_key]

                result = func(*args, **kwargs)

                cache["data"][cache_key] = result
                with cache_file.open("w") as f:
                    yaml.safe_dump(cache, f)

                return result

            return wrapper

        return decorator

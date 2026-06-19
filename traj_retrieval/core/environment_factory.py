# traj_retrieval/core/environment_factory.py
# Central registry and factory for environment handlers

from typing import Dict, Type, List, Optional
from .base_handler import EnvironmentHandler


class EnvironmentFactory:
    """
    Central registry and factory for environment handlers.

    Uses lazy imports to avoid loading unnecessary dependencies.
    Handlers are only imported when requested via create_handler().

    This factory makes it easy to add new environments without modifying existing code.
    Simply create a new handler that implements EnvironmentHandler and register it.

    Example usage:
        # Create a handler (imports only what's needed)
        handler = EnvironmentFactory.create_handler("alfworld")

        # Register a custom handler
        EnvironmentFactory.register_handler("my_env", MyEnvHandler)

        # Check available environments
        envs = EnvironmentFactory.get_available_environments()
    """

    # Registry of environment name -> handler class (lazy loaded)
    _handlers: Dict[str, Type[EnvironmentHandler]] = {}

    # Registry of environment name -> import path (module, class_name)
    _handler_imports: Dict[str, tuple] = {
        "alfworld": (".alfworld_handler", "AlfWorldHandler"),
        "webarena": (".webarena_handler", "WebArenaHandler"),
    }

    @classmethod
    def _lazy_import_handler(
        cls, environment_name: str
    ) -> Optional[Type[EnvironmentHandler]]:
        """
        Lazily import a handler class only when needed.

        Args:
            environment_name: Name of the environment

        Returns:
            Handler class if import succeeds, None otherwise
        """
        env_name = environment_name.lower().strip()

        # Check if already loaded
        if env_name in cls._handlers:
            return cls._handlers[env_name]

        # Check if we have import info for this environment
        if env_name not in cls._handler_imports:
            return None

        module_path, class_name = cls._handler_imports[env_name]

        # Try to import the handler
        try:
            from importlib import import_module

            module = import_module(module_path, package=__package__)
            handler_class = getattr(module, class_name)

            # Validate that it implements EnvironmentHandler
            if not issubclass(handler_class, EnvironmentHandler):
                print(
                    f"[Factory] Warning: {class_name} does not implement EnvironmentHandler"
                )
                return None

            # Cache the loaded handler
            cls._handlers[env_name] = handler_class
            return handler_class

        except ImportError as e:
            print(f"[Factory] Environment '{environment_name}' not available: {e}")
            return None
        except Exception as e:
            print(f"[Factory] Failed to load handler for '{environment_name}': {e}")
            return None

    @classmethod
    def create_handler(
        cls, environment_name: str, **handler_kwargs
    ) -> EnvironmentHandler:
        """
        Create an environment handler instance by name with configuration.

        Lazily imports the handler class only when needed.

        Args:
            environment_name: Name of the environment (case-insensitive)
            **handler_kwargs: Environment-specific configuration parameters
                             passed to the handler constructor

        Returns:
            Instance of the requested EnvironmentHandler

        Raises:
            ValueError: If environment is not supported or import fails

        Example:
            >>> # Create an ALFWorld handler with default config
            >>> handler = EnvironmentFactory.create_handler("alfworld")
            >>>
            >>> # Create a WebArena handler with custom config
            >>> handler = EnvironmentFactory.create_handler(
            ...     "webarena",
            ...     max_steps=30,
            ... )
        """
        # Try to load the handler (lazy import)
        handler_class = cls._lazy_import_handler(environment_name)

        if handler_class is None:
            available = ", ".join(sorted(cls._handler_imports.keys()))
            raise ValueError(
                f"Failed to load environment: '{environment_name}'. "
                f"Registered environments: [{available}]. "
                f"Make sure the environment and its dependencies are installed."
            )

        # Create instance with provided configuration
        return handler_class(**handler_kwargs)

    @classmethod
    def register_handler(
        cls,
        environment_name: str,
        handler_class: Type[EnvironmentHandler],
        override: bool = False,
    ):
        """
        Register a new environment handler with an already-loaded class.

        This allows for plugin-style extensions where new environments can be
        registered at runtime without modifying the core code.

        Note: The handler class is cached immediately (not lazy-loaded).
        For lazy loading, use register_handler_import() instead.

        Args:
            environment_name: Name to register the handler under (case-insensitive)
            handler_class: Class that implements EnvironmentHandler interface
            override: If True, allows overriding existing handlers (default: False)

        Raises:
            ValueError: If handler already exists and override=False
            TypeError: If handler_class doesn't implement EnvironmentHandler

        Example:
            >>> class MyEnvHandler(EnvironmentHandler):
            ...     # ... implementation ...
            ...     pass
            >>> EnvironmentFactory.register_handler("myenv", MyEnvHandler)
        """
        # Validate that handler_class implements EnvironmentHandler
        if not issubclass(handler_class, EnvironmentHandler):
            raise TypeError(
                f"Handler class must implement EnvironmentHandler interface. "
                f"Got: {handler_class}"
            )

        env_name = environment_name.lower().strip()

        # Check for conflicts
        if env_name in cls._handlers and not override:
            raise ValueError(
                f"Environment '{environment_name}' is already registered. "
                f"Use override=True to replace it."
            )

        # Register the loaded class directly
        cls._handlers[env_name] = handler_class
        print(f"[Factory] Registered environment handler: '{environment_name}'")

    @classmethod
    def register_handler_import(
        cls,
        environment_name: str,
        module_path: str,
        class_name: str,
        override: bool = False,
    ):
        """
        Register a new environment handler by import path (for lazy loading).

        The handler will only be imported when first requested via create_handler().

        Args:
            environment_name: Name to register the handler under (case-insensitive)
            module_path: Python module path (e.g., ".my_handler" or "mypackage.handlers")
            class_name: Name of the handler class in the module
            override: If True, allows overriding existing handlers (default: False)

        Raises:
            ValueError: If handler already exists and override=False

        Example:
            >>> EnvironmentFactory.register_handler_import(
            ...     "myenv", ".my_env_handler", "MyEnvHandler"
            ... )
        """
        env_name = environment_name.lower().strip()

        # Check for conflicts
        if env_name in cls._handler_imports and not override:
            raise ValueError(
                f"Environment '{environment_name}' is already registered. "
                f"Use override=True to replace it."
            )

        cls._handler_imports[env_name] = (module_path, class_name)
        print(
            f"[Factory] Registered environment import: '{environment_name}' -> {module_path}.{class_name}"
        )

    @classmethod
    def get_available_environments(cls) -> List[str]:
        """
        Get list of available environment names.

        Returns all registered environments (from _handler_imports),
        not just the ones that have been loaded.

        Returns:
            Sorted list of registered environment names

        Example:
            >>> envs = EnvironmentFactory.get_available_environments()
            >>> print(envs)
            ['alfworld', 'webarena']
        """
        return sorted(cls._handler_imports.keys())

    @classmethod
    def is_environment_supported(cls, environment_name: str) -> bool:
        """
        Check if an environment is supported.

        Checks if the environment is registered in _handler_imports.
        Does not verify if dependencies are installed.

        Args:
            environment_name: Name to check (case-insensitive)

        Returns:
            True if environment is registered, False otherwise

        Example:
            >>> EnvironmentFactory.is_environment_supported("alfworld")
            True
            >>> EnvironmentFactory.is_environment_supported("unknown")
            False
        """
        return environment_name.lower().strip() in cls._handler_imports

    @classmethod
    def get_handler_class(cls, environment_name: str) -> Type[EnvironmentHandler]:
        """
        Get the handler class for an environment without instantiating it.

        Lazily imports the handler class if not already loaded.
        Useful for inspection or advanced use cases.

        Args:
            environment_name: Name of the environment (case-insensitive)

        Returns:
            The handler class (not an instance)

        Raises:
            ValueError: If environment is not supported or import fails
        """
        # Try to load the handler (lazy import)
        handler_class = cls._lazy_import_handler(environment_name)

        if handler_class is None:
            available = ", ".join(sorted(cls._handler_imports.keys()))
            raise ValueError(
                f"Failed to load environment: '{environment_name}'. "
                f"Registered environments: [{available}]. "
                f"Make sure the environment and its dependencies are installed."
            )

        return handler_class

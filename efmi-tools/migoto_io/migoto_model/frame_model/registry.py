from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from migoto_model.frame_model.commands import CommandCall


COMMAND_REGISTRY: dict[str, Type["CommandCall"]] = {}


def register_command(name: str):
    def wrapper(cls):
        if name in COMMAND_REGISTRY:
            raise ValueError(f"Command {name} already registered")
        COMMAND_REGISTRY[name] = cls
        return cls

    return wrapper

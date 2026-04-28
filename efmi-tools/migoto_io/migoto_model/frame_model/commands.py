from dataclasses import dataclass, field, fields
from typing import ClassVar
from typing import get_args
from typing import TYPE_CHECKING

from ..log_model.log_model import DumpedBinding, DumpedCommand
from ..types import ShaderType
from ..helpers import AutoArgsMixin

from .registry import COMMAND_REGISTRY

if TYPE_CHECKING:
    from .frame_model import DumpModel
    from .calls import ShaderCall


@dataclass
class CommandBinding(AutoArgsMixin):
    raw_binding: DumpedBinding

    @staticmethod
    def parse_binding(line: str) -> dict[str, str]:
        parts = line.split(":", 1)

        if len(parts) != 2:
            raise ValueError("Prefix is not found!")

        prefix = parts[0].strip()
        arguments = parts[1].strip()

        kwargs = CommandBinding.parse_kv_pairs(arguments)
        kwargs["prefix"] = prefix

        return kwargs

    @classmethod
    def from_arguments(cls, raw_binding: DumpedBinding, arguments: dict[str, str]) -> "CommandBinding":
        kwargs = cls.field_kwargs_from_dict(arguments)
        kwargs["raw_binding"] = raw_binding
        binding = cls(**kwargs)
        return binding

    @classmethod
    def from_dumped_binding(cls, raw_binding: DumpedBinding) -> "CommandBinding":
        binding_arguments = cls.parse_binding(raw_binding.line)
        obj: "CommandBinding" = cls.from_arguments(raw_binding=raw_binding, arguments=binding_arguments)
        return obj


    # def __repr__(self):
    #     if type(self) is CommandBinding:
    #         return self.raw_binding.line
    #
    #     parts = []
    #     for f in fields(self):
    #         if f.name == "raw_binding":
    #             continue
    #         parts.append(f"{f.name}={getattr(self, f.name)}")
    #
    #     return ", ".join(parts)


@dataclass
class ParseCommandCallConfig:
    skip_commands: set[str] = field(default_factory=set)
    skip_stage_commands: set[str] = field(default_factory=set)


@dataclass
class CommandCall(AutoArgsMixin):
    bindings: list[CommandBinding]
    raw_command: DumpedCommand
    shader_type: ShaderType = field(init=False)

    _binding_type_cache: ClassVar[dict | None] = None
    _repr_fields_cache: ClassVar[list[str] | None] = None
    _fast_repr = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._binding_type_cache = None
        cls._repr_fields_cache = None

    @classmethod
    def binding_type(cls):
        if cls._binding_type_cache is None:
            for f in fields(cls):
                if f.name == "bindings":
                    args = get_args(f.type)
                    cls._binding_type_cache = args[0] if args else CommandBinding
                    break
        return cls._binding_type_cache

    @classmethod
    def from_arguments(cls, raw_command: DumpedCommand, arguments: dict[str, str]) -> "CommandCall":
        kwargs = cls.field_kwargs_from_dict(arguments)
        kwargs["raw_command"] = raw_command
        kwargs["bindings"] = [cls.binding_type().from_dumped_binding(b) for b in raw_command.bindings]

        call = cls(**kwargs)

        if not hasattr(call, "shader_type"):
            call.shader_type = ShaderType.Any

        return call

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        pass

    @classmethod
    def from_dumped_command(
        cls,
        raw_command: DumpedCommand,
        cfg: ParseCommandCallConfig,
    ) -> "CommandCall | None" :

        command_name, command_arguments = cls.parse_command(raw_command.line)

        if command_name in cfg.skip_commands:
            return None
        if command_name[2:] in cfg.skip_stage_commands:
            return None

        handler = COMMAND_REGISTRY.get(command_name)
        if handler is None:
            call = cls(
                bindings=[],
                raw_command=raw_command,
            )
            call.shader_type = ShaderType.Any
            # print(f'[{raw_command.line_id+1}]: Unknown command {command_name}: {raw_command.line}')
            return call

        if not command_name.startswith("3DMigoto"):
            command_arguments = cls.parse_arguments(command_arguments)

        obj: "CommandCall" = handler.from_arguments(raw_command=raw_command, arguments=command_arguments)

        return obj

    @staticmethod
    def parse_command(line: str) -> tuple[str, str]:
        line = line.strip()

        if line.startswith("3DMigoto Dumping"):
            command = "3DMigoto Dumping"
            arguments = line[17:]

        else:
            parts = line.split("(", 1)
            if len(parts) != 2:
                raise ValueError("Command arguments must start with `(`!")

            command = parts[0].strip()
            arguments = parts[1].strip()

        return command, arguments

    @staticmethod
    def parse_arguments(arguments: str) -> dict[str, str]:
        parts = arguments.split(")", 1)
        if len(parts) != 2:
            raise ValueError("Command arguments must end with `)`!")

        arguments_str = parts[0].strip()
        properties_str = parts[1].strip()

        arguments = {}

        if arguments_str:
            arguments.update(CommandCall.parse_kv_pairs(arguments_str, pair_separator=", ", value_separator=":"))

        if properties_str:
            arguments.update(CommandCall.parse_kv_pairs(properties_str))

        return arguments

    # def __repr__(self):
    #     if self._fast_repr or type(self) is CommandCall:
    #         return self.raw_command.line
    #
    #     cls = type(self)
    #     if cls._repr_fields_cache is None:
    #         cls._repr_fields_cache = [
    #             f.name for f in fields(cls)
    #             if f.name not in ("bindings", "raw_command")
    #         ]
    #
    #     args = [f"{name}={getattr(self, name)}" for name in cls._repr_fields_cache]
    #     return ", ".join(args)

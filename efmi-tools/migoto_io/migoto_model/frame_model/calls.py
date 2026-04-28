
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .frame_model import DumpModel


from ..log_model.log_model import DumpedCall
from ..types import ShaderType, ResourceSlot

from .resources import Resource, ResourceStorage
from .commands import CommandCall, ParseCommandCallConfig


@dataclass
class Shader:
    hash: str
    pointer: str
    type: ShaderType


@dataclass
class ParseShaderCallConfig:
    command_config: ParseCommandCallConfig = field(default_factory=lambda: ParseCommandCallConfig())


@dataclass
class ShaderCall:
    id: int
    shaders: list[Shader]
    commands: list[CommandCall]
    resources: ResourceStorage | None = None  # Resources explicitly set by commands of this call
    model_resources: ResourceStorage | None = None  # Both explicitly and implicitly set resources
    draw_call: CommandCall | None = None

    @classmethod
    def from_dumped_call(
        cls,
        raw_call: DumpedCall,
        cfg: ParseShaderCallConfig,
    ) -> "ShaderCall":
        commands = []
        for raw_command in raw_call.commands:
            command = CommandCall.from_dumped_command(
                raw_command=raw_command,
                cfg=cfg.command_config,
            )
            if command:
                commands.append(command)
        return cls(
            id=raw_call.id,
            shaders=[],
            commands=commands,
        )

    def execute_commands(self, dump_model: "DumpModel"):
        self.resources=ResourceStorage()
        self.model_resources=ResourceStorage()
        for command in self.commands:
            command.execute(dump_model=dump_model, shader_call=self)
        # Snapshot resource slots state
        for slot_index in dump_model.current_resources._slot_type_index.values():
            for slot, resource in slot_index.items():
                self.model_resources.add(slot, resource)

    def set_resource(self, slot: ResourceSlot, resource: Resource):
        self.resources.add(slot, resource)

    def clear_resource(self, resource: ResourceSlot | Resource):
        self.resources.remove(resource)

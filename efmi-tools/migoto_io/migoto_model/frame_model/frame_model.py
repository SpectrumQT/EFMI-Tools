
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from ..log_model.log_model import FrameDumpLog
from ..types import ResourceSlot

from .calls import ShaderCall, ParseShaderCallConfig
from .resources import Resource, ResourceStorage

# Import and register all CommandCall's
from . import api_calls


@dataclass
class ParseDumpModelConfig:
    dump_path: Path
    track_resource_usage: bool = False
    shader_call_config: ParseShaderCallConfig = field(default_factory=lambda: ParseShaderCallConfig())


@dataclass
class DumpModel:
    cfg: ParseDumpModelConfig
    calls: list[ShaderCall]
    resources: dict[str, Resource]
    current_resources: ResourceStorage | None = None  # Current state of resource slots
    current_call: ShaderCall | None = None

    @classmethod
    def from_frame_dump_log(
        cls,
        log: FrameDumpLog,
        cfg: ParseDumpModelConfig,
    ) -> "DumpModel":
        calls = []
        for raw_call in log.calls:
            call = ShaderCall.from_dumped_call(
                raw_call=raw_call,
                cfg=cfg.shader_call_config,
            )
            if call:
                calls.append(call)
        return cls(
            cfg=cfg,
            calls=calls,
            resources={},
            current_resources=ResourceStorage(),
        )

    def execute_commands(self):
        self.resources = {}
        self.current_resources = ResourceStorage()
        for shader_call in self.calls:
            self.current_call = shader_call
            shader_call.execute_commands(dump_model=self)

    def register_resource(self, resource: Resource) -> Resource:
        if self.cfg.track_resource_usage:
            registered_resource = self.resources.get(resource.pointer, None)

            if registered_resource is None:
                resource.usage = defaultdict(list)
            else:
                resource.usage = defaultdict(list, {
                    cid: slots_list.copy() for cid, slots_list in registered_resource.usage.items()
                })

        self.resources[resource.pointer] = resource

        return resource

    def set_resource(self, slot: ResourceSlot, resource: Resource):
        registered_resource = self.register_resource(resource)

        if self.cfg.track_resource_usage:
            resource.usage[self.current_call.id].append(slot)

        self.current_resources.add(slot, registered_resource)
        self.current_call.set_resource(slot, registered_resource)

    def clear_resource(self, resource: ResourceSlot | Resource):
        self.current_resources.remove(resource)
        self.current_call.clear_resource(resource)

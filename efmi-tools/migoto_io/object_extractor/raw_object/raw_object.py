from dataclasses import dataclass, field

from ...migoto_model.types import SlotType, ShaderType, ResourceSlot
from ...migoto_model.frame_model.calls import ShaderCall
from ...migoto_model.frame_model.resources import Resource


@dataclass
class RawComponent:
    vertex_offset: int
    vertex_count: int
    shader_calls: list[ShaderCall] = field(default_factory=list)

    def get_resources(
        self,
        slot_type: SlotType,
        shader_type: ShaderType | None = None,
        skip_implicit: bool = True
    ) -> dict[ResourceSlot, list[Resource]]:

        resources: dict[ResourceSlot, list[Resource]] = {}
        for shader_call in self.shader_calls:
            if skip_implicit:
                resources_storage = shader_call.resources
            else:
                resources_storage = shader_call.model_resources
            for slot, resource in resources_storage.get_slot_index(slot_type).items():
                if shader_type is not None and shader_type != slot.shader_type:
                    continue
                slot_resources = resources.get(slot, [])
                if not slot_resources:
                    resources[slot] = slot_resources
                slot_resources.append(resource)
        return resources


@dataclass
class RawObject:
    id: str
    components: dict[tuple[str, int, int], RawComponent] = field(default_factory=dict)

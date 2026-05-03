from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from typing import TYPE_CHECKING

from ...types import MigotoResourceType, ShaderType, SlotType, ResourceSlot
from ...log_model.log_model import DumpedCommand
from ...filename_descriptors import MigotoResourceDescriptor, MigotoBufferDescriptor, MigotoTextureDescriptor, MigotoIndexBufferDescriptor, MigotoVertexBufferDescriptor
from ..registry import register_command
from ..commands import CommandCall
from ..resources import MigotoBuffer

if TYPE_CHECKING:
    from ..frame_model import DumpModel
    from ..calls import ShaderCall


# 3DMigoto Dumping Buffer C:\Dump\000189-vs-cb0=a517561d-vs=833324977c629596-ps=8a2793b043be184b.buf -> C:\Dump\deduped\64fdcd33.buf
# 3DMigoto Dumping Texture2D C:\Dump\000189-ps-t2=a62fc2b3-vs=833324977c629596-ps=8a2793b043be184b.jpg -> C:\Dump\deduped\81fcdb72-R16G16B16A16_FLOAT.jpg
@register_command("3DMigoto Dumping")
@dataclass(repr=False)
class MigotoDumpFile(CommandCall):
    resource_type: MigotoResourceType = field(metadata={"arg": "resource_type"})
    resource_path: Path = field(metadata={"arg": "resource_path"})
    deduped_path: Path = field(metadata={"arg": "deduped_path"})

    _SLOT_FILENAME_TYPES: ClassVar[dict[SlotType, type[MigotoResourceDescriptor]]] = {
        SlotType.ConstantBuffer: MigotoResourceDescriptor,
        SlotType.Texture: MigotoTextureDescriptor,
        SlotType.VertexBuffer: MigotoVertexBufferDescriptor,
        SlotType.IndexBuffer: MigotoIndexBufferDescriptor,
        SlotType.RenderTarget: MigotoTextureDescriptor,
        SlotType.UAV: MigotoResourceDescriptor,
    }

    @classmethod
    def from_arguments(cls, raw_command: DumpedCommand, arguments: str) -> "CommandCall":
        parts = arguments.split("->", 1)
        if len(parts) != 2:
            raise ValueError("3DMigoto Dumping command must contain `->`!")

        arguments = {'deduped_path': parts[1].strip()}

        parts = parts[0].split(" ", 1)

        if len(parts) != 2:
            raise ValueError("3DMigoto Dumping command must have type and path separated with ` `!")

        arguments['resource_type'] = parts[0].strip()
        arguments['resource_path'] = parts[1].strip()

        return super().from_arguments(raw_command=raw_command, arguments=arguments)

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        # slot = ResourceSlot(self.shader_type, SlotType.IndexBuffer, 0)
        # resource = IndexBuffer(
        #     pointer=self.pointer,
        #     hash=self.hash,
        #     byte_offset=self.byte_offset,
        #     format=self.format,
        # )
        # dump_model.set_resource(slot, resource)

        usage_descriptor = MigotoBufferDescriptor.from_filename(self.resource_path)

        slot = ResourceSlot(usage_descriptor.shader_type or ShaderType.Any, usage_descriptor.slot_type, usage_descriptor.slot_id or 0)
        resource = shader_call.resources.get_by_slot(slot)

        if resource is None:
            resource = dump_model.current_resources.get_by_slot(slot)

        # Always write last used descriptors and paths to main resource
        # View should be set in slot only when offset and size can be calculated
        # - As of now, view fmt is available only on "3DMigoto Dumping Buffer" for .txt
        if resource.parent:
            resource = resource.parent

        # if resource.parent:
        #     if not usage_descriptor.original_hash and resource.parent.hash == usage_descriptor.resource_hash:
        #         return
        #     if self.resource_path.suffix == ".buf":
        #         return

        resource.usage_descriptor = usage_descriptor

        if self.deduped_path.suffix != ".buf":
            data_descriptor = self._SLOT_FILENAME_TYPES[usage_descriptor.slot_type].from_filename(self.deduped_path)
            # if resource.hash != data_descriptor.resource_hash:
            #     raise ValueError
            resource.data_descriptor = data_descriptor

        if usage_descriptor.resource_hash is None:
            shader_call.has_unknown_resource = True
            if self.deduped_path.suffix != ".buf":
                usage_descriptor.resource_hash = f"UNKNOWN_{data_descriptor.resource_hash}"
            else:
                usage_descriptor.resource_hash = f"UNKNOWN_{self.deduped_path.stem}"
            resource.hash = usage_descriptor.resource_hash

        if self.resource_path.suffix == ".txt":
            resource.txt_path = dump_model.cfg.dump_path / self.resource_path.name
            resource.txt_path_deduped = dump_model.cfg.dump_path / 'deduped' / self.deduped_path.name
        else:
            resource.bin_path = dump_model.cfg.dump_path / self.resource_path.name
            resource.bin_path_deduped = dump_model.cfg.dump_path / 'deduped' / self.deduped_path.name

        if resource.hash is None or resource.hash == "None":
            return
        
        if isinstance(resource, MigotoBuffer) and resource.data_descriptor is not None:

            resource.load_format()

            if usage_descriptor.original_hash == resource.hash:
                # Should never happen
                if resource.hash == usage_descriptor.resource_hash:
                    raise ValueError

                resource_view = resource.get_view()
                dump_model.set_resource(slot, resource_view)

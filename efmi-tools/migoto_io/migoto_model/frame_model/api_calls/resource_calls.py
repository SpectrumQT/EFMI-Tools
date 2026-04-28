from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...types import ShaderType, SlotType, ResourceSlot, DXGI_FORMAT
from ..resources import Resource, ConstantBuffer, VertexBuffer, IndexBuffer
from ..registry import register_command
from ..commands import CommandCall, CommandBinding

if TYPE_CHECKING:
    from ..frame_model import DumpModel
    from ..calls import ShaderCall


@dataclass(repr=False)
class ResourceSlotBinding(CommandBinding):
    slot_id: int = field(metadata={"arg": "prefix"})
    resource: str = field(metadata={"arg": "resource"})
    hash: str = field(metadata={"arg": "hash"})


@dataclass(repr=False)
class ConstantBufferSlotBinding(ResourceSlotBinding):
    first_constant: int | None = field(metadata={"arg": "first_constant"})
    num_constants: int | None = field(metadata={"arg": "num_constants"})


@dataclass(repr=False)
class SetConstantBuffers(CommandCall):
    bindings: list[ConstantBufferSlotBinding]
    start_slot: int = field(metadata={"arg": "StartSlot"})
    num_buffers: int = field(metadata={"arg": "NumBuffers"})

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        for binding in self.bindings:
            slot = ResourceSlot(self.shader_type, SlotType.ConstantBuffer, binding.slot_id)
            resource = ConstantBuffer(
                pointer=binding.resource,
                hash=binding.hash,
                first_constant=binding.first_constant,
                num_constants=binding.num_constants,
            )
            dump_model.set_resource(slot, resource)


# CSSetConstantBuffers1(StartSlot:0, NumBuffers:1, ppConstantBuffers:0x000000007B42C940, pFirstConstant:0x000000007B42D670, pNumConstants:0x000000007B42D6A8)
#        0: resource=0x0000001D932651A0 hash=a517561d first_constant=16640 num_constants=16
@register_command("CSSetConstantBuffers")
@register_command("CSSetConstantBuffers1")
@dataclass(repr=False)
class CSSetConstantBuffers(SetConstantBuffers):
    shader_type = ShaderType.Compute


# VSSetConstantBuffers1(StartSlot:0, NumBuffers:1, ppConstantBuffers:0x000000007B42C8F0, pFirstConstant:0x000000007B42CBA0, pNumConstants:0x000000007B42CBD8)
#        0: resource=0x0000001D932651A0 hash=a517561d first_constant=9376 num_constants=16
@register_command("VSSetConstantBuffers")
@register_command("VSSetConstantBuffers1")
@dataclass(repr=False)
class VSSetConstantBuffers(SetConstantBuffers):
    shader_type = ShaderType.Vertex


# PSSetConstantBuffers1(StartSlot:0, NumBuffers:1, ppConstantBuffers:0x000000007B42C8F0, pFirstConstant:0x000000007B42D108, pNumConstants:0x000000007B42D140)
#        0: resource=0x0000001D932651A0 hash=a517561d first_constant=4768 num_constants=288
@register_command("PSSetConstantBuffers")
@register_command("PSSetConstantBuffers1")
@dataclass(repr=False)
class PSSetConstantBuffers(SetConstantBuffers):
    shader_type = ShaderType.Pixel


# IASetVertexBuffers(StartSlot:0, NumBuffers:1, ppVertexBuffers:0x000000007B42DF98, pStrides:0x00000001FB0011DC, pOffsets:0x000000007B42DFD8)
#        0: resource=0x0000001D17A6CEA0 hash=122c46be
@register_command("IASetVertexBuffers")
@dataclass(repr=False)
class SetVertexBuffers(CommandCall):
    bindings: list[ResourceSlotBinding]

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        for binding in self.bindings:
            slot = ResourceSlot(self.shader_type, SlotType.VertexBuffer, binding.slot_id)
            resource = VertexBuffer(
                pointer=binding.resource,
                hash=binding.hash,
            )
            dump_model.set_resource(slot, resource)


# IASetIndexBuffer(pIndexBuffer:0x0000001D17A705A0, Format:57, Offset:0) hash=78a3d343
@register_command("IASetIndexBuffer")
@dataclass(repr=False)
class SetIndexBuffer(CommandCall):
    pointer: str = field(metadata={"arg": "pIndexBuffer"})
    hash: str = field(metadata={"arg": "hash"})
    byte_offset: int = field(metadata={"arg": "Offset"})
    format: DXGI_FORMAT = field(metadata={"arg": "Format"})

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        slot = ResourceSlot(self.shader_type, SlotType.IndexBuffer, 0)
        resource = IndexBuffer(
            pointer=self.pointer,
            hash=self.hash,
            byte_offset=self.byte_offset,
            format=self.format,
        )
        dump_model.set_resource(slot, resource)


@dataclass(repr=False)
class SetShaderResources(CommandCall):
    bindings: list[ResourceSlotBinding]
    start_slot: int = field(metadata={"arg": "StartSlot"})
    num_views: int = field(metadata={"arg": "NumViews"})

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        for binding in self.bindings:
            slot = ResourceSlot(self.shader_type, SlotType.Texture, binding.slot_id)
            resource = Resource(
                pointer=binding.resource,
                hash=binding.hash,
            )
            dump_model.set_resource(slot, resource)


# CSSetShaderResources(StartSlot:0, NumViews:3, ppShaderResourceViews:0x000000007B42D178)
#        0: view=0x0000000129761E28 resource=0x0000000127BA21E0 hash=b7ff7a6e
@register_command("CSSetShaderResources")
@dataclass(repr=False)
class CSSetShaderResources(SetShaderResources):
    shader_type = ShaderType.Compute


# VSSetShaderResources(StartSlot:0, NumViews:3, ppShaderResourceViews:0x000000007B42D178)
#        0: view=0x0000000129761E28 resource=0x0000000127BA21E0 hash=b7ff7a6e
@register_command("VSSetShaderResources")
@dataclass(repr=False)
class VSSetShaderResources(SetShaderResources):
    shader_type = ShaderType.Vertex


# PSSetShaderResources(StartSlot:0, NumViews:3, ppShaderResourceViews:0x000000007B42D178)
#        0: view=0x0000000129761E28 resource=0x0000000127BA21E0 hash=b7ff7a6e
@register_command("PSSetShaderResources")
@dataclass(repr=False)
class PSSetShaderResources(SetShaderResources):
    shader_type = ShaderType.Pixel


@dataclass(repr=False)
class SetUnorderedAccessViews(CommandCall):
    bindings: list[ResourceSlotBinding]
    start_slot: int = field(metadata={"arg": "StartSlot"})
    num_uavs: int = field(metadata={"arg": "NumUAVs"})

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        for binding in self.bindings:
            slot = ResourceSlot(self.shader_type, SlotType.UAV, binding.slot_id)
            resource = Resource(
                pointer=binding.resource,
                hash=binding.hash,
            )
            dump_model.set_resource(slot, resource)


# CSSetUnorderedAccessViews(StartSlot:0, NumUAVs:1, ppUnorderedAccessViews:0x000000006CD3FD98, pUAVInitialCounts:0x000000006CD3FD80)
#        0: view=0x0000000086515ED0 resource=0x00000000D3213678 hash=ad31580f
@register_command("CSSetUnorderedAccessViews")
@dataclass(repr=False)
class CSSetUnorderedAccessViews(SetUnorderedAccessViews):
    shader_type = ShaderType.Compute


@dataclass(repr=False)
class RenderTargetSlotBinding(CommandBinding):
    slot_id: str = field(metadata={"arg": "prefix"})
    resource: str = field(metadata={"arg": "resource"})
    hash: str = field(metadata={"arg": "hash"})


# OMSetRenderTargets(NumViews:1, ppRenderTargetViews:0x000000007B42C8F0, pDepthStencilView:0x0000001D0D05B120)
#        0: view=0x0000001D4D9AC520 resource=0x0000001D4D938AA0 hash=d9f11fd9
#        D: view=0x0000001D0D05B120 resource=0x0000001D4D939020 hash=c5057d7e
@register_command("OMSetRenderTargets")
@dataclass(repr=False)
class OMSetRenderTargets(CommandCall):
    bindings: list[RenderTargetSlotBinding]
    num_views: int = field(metadata={"arg": "NumViews"})

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        for binding in self.bindings:
            # Skip depth stencil resource binding
            if binding.slot_id == "D":
                continue

            slot = ResourceSlot(self.shader_type, SlotType.RenderTarget, int(binding.slot_id))
            resource = Resource(
                pointer=binding.resource,
                hash=binding.hash,
            )
            dump_model.set_resource(slot, resource)


# OMSetRenderTargetsAndUnorderedAccessViews(NumRTVs:-1, ppRenderTargetViews:0x0000000000000000, pDepthStencilView:0x0000000000000000, UAVStartSlot:5, NumUAVs:1, ppUnorderedAccessViews:0x000000007B42DB90, pUAVInitialCounts:0x00007FF90A8920D0)
#        5: view=0x0000001BCC633BB8 resource=0x0000001BEF631020 hash=17048a49
@register_command("OMSetRenderTargetsAndUnorderedAccessViews")
@dataclass(repr=False)
class OMSetRenderTargetsAndUnorderedAccessViews(CommandCall):
    bindings: list[RenderTargetSlotBinding]
    num_rtvs: int = field(metadata={"arg": "NumRTVs"})
    start_slot: int = field(metadata={"arg": "UAVStartSlot"})
    num_uavs: int = field(metadata={"arg": "NumUAVs"})

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        for binding in self.bindings:
            # Skip depth stencil resource binding
            if binding.slot_id == "D":
                continue

            slot = ResourceSlot(self.shader_type, SlotType.UAV, int(binding.slot_id))
            resource = Resource(
                pointer=binding.resource,
                hash=binding.hash,
            )
            dump_model.set_resource(slot, resource)
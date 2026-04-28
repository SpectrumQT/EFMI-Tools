from dataclasses import dataclass
from enum import Enum, EnumMeta


class ShaderType(Enum):
    Any = 'any'
    Compute = 'cs'
    Pixel = 'ps'
    Vertex = 'vs'
    Geometry = 'gs'
    Hull = 'hs'
    Domain = 'ds'


class SlotType(Enum):
    ConstantBuffer = 'cb'
    Texture = 't'
    VertexBuffer = 'vb'
    IndexBuffer = 'ib'
    RenderTarget = 'o'
    UAV = 'u'


OUTPUT_SLOT_TYPES: list[SlotType] = [SlotType.UAV, SlotType.RenderTarget]


@dataclass(frozen=True, slots=True)
class ResourceSlot:
    shader_type: ShaderType
    slot_type: SlotType
    slot_id: int

    def __post_init__(self):
        # Remove shader context for output slots like 3dmigoto does
        if self.slot_type in OUTPUT_SLOT_TYPES:
            object.__setattr__(self, "shader_type", ShaderType.Any)

    def __str__(self):
        shader_type = f"{self.shader_type.value}-" if self.shader_type != ShaderType.Any else ""
        slot_id = self.slot_id if self.slot_type != SlotType.IndexBuffer else ""
        return f'{shader_type}{self.slot_type.value}{slot_id}'

    # def __repr__(self):
    #     return self.__str__()

    @staticmethod
    def split_slot(slot: str) -> tuple[SlotType, int]:
        slot_type_str = slot.rstrip("0123456789")
        slot_type = SlotType(slot_type_str)
        slot_id = int(slot[len(slot_type_str):] or 0)
        return slot_type, slot_id

    @classmethod
    def from_string(cls, slot_str) -> "ResourceSlot | None":
        parts = slot_str.split("-", 1)
        slot = parts[-1]
        slot_type, slot_id = cls.split_slot(slot)
        shader_type = ShaderType(parts[0]) if len(parts) == 2 else ShaderType.Any
        return cls(
            shader_type=shader_type,
            slot_type=slot_type,
            slot_id=slot_id,
        )


class ContaminationType(Enum):
    Region = 'S'
    Copy = 'C'
    Update = 'U'
    Map = 'M'


class Topology(Enum):
    Undefined = 'undefined'
    PointList = 'pointlist'
    LineList = 'linelist'
    LineStrip = 'linestrip'
    TriangleList = 'trianglelist'
    TriangleStrip = 'trianglestrip'
    LineListAdj = 'linelist_adj'
    LineStripAdj = 'linestrip_adj'
    TriangleListAdj = 'trianglelist_adj'
    TriangleStripAdj = 'trianglestrip_adj'

    # Patchlists
    ControlPointPatchList1 = '1_control_point_patchlist'
    ControlPointPatchList2 = '2_control_point_patchlist'
    ControlPointPatchList3 = '3_control_point_patchlist'
    ControlPointPatchList4 = '4_control_point_patchlist'
    ControlPointPatchList5 = '5_control_point_patchlist'
    ControlPointPatchList6 = '6_control_point_patchlist'
    ControlPointPatchList7 = '7_control_point_patchlist'
    ControlPointPatchList8 = '8_control_point_patchlist'
    ControlPointPatchList9 = '9_control_point_patchlist'
    ControlPointPatchList10 = '10_control_point_patchlist'
    ControlPointPatchList11 = '11_control_point_patchlist'
    ControlPointPatchList12 = '12_control_point_patchlist'
    ControlPointPatchList13 = '13_control_point_patchlist'
    ControlPointPatchList14 = '14_control_point_patchlist'
    ControlPointPatchList15 = '15_control_point_patchlist'
    ControlPointPatchList16 = '16_control_point_patchlist'
    ControlPointPatchList17 = '17_control_point_patchlist'
    ControlPointPatchList18 = '18_control_point_patchlist'
    ControlPointPatchList19 = '19_control_point_patchlist'
    ControlPointPatchList20 = '20_control_point_patchlist'
    ControlPointPatchList21 = '21_control_point_patchlist'
    ControlPointPatchList22 = '22_control_point_patchlist'
    ControlPointPatchList23 = '23_control_point_patchlist'
    ControlPointPatchList24 = '24_control_point_patchlist'
    ControlPointPatchList25 = '25_control_point_patchlist'
    ControlPointPatchList26 = '26_control_point_patchlist'
    ControlPointPatchList27 = '27_control_point_patchlist'
    ControlPointPatchList28 = '28_control_point_patchlist'
    ControlPointPatchList29 = '29_control_point_patchlist'
    ControlPointPatchList30 = '30_control_point_patchlist'
    ControlPointPatchList31 = '31_control_point_patchlist'
    ControlPointPatchList32 = '32_control_point_patchlist'

    def __str__(self):
        return f'{self.value}'

    # def __repr__(self):
    #     return f'{self.value}'


class MigotoResourceType(Enum):
    Buffer = "Buffer"
    Texture2D = "Texture2D"


class DXGI_FORMAT_META(EnumMeta):
    def __call__(cls, value, *args, **kwargs):
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        return super().__call__(value, *args, **kwargs)


class DXGI_FORMAT(Enum, metaclass=DXGI_FORMAT_META):
    DXGI_FORMAT_UNKNOWN	                                = 0
    DXGI_FORMAT_R32G32B32A32_TYPELESS                   = 1
    DXGI_FORMAT_R32G32B32A32_FLOAT                      = 2
    DXGI_FORMAT_R32G32B32A32_UINT                       = 3
    DXGI_FORMAT_R32G32B32A32_SINT                       = 4
    DXGI_FORMAT_R32G32B32_TYPELESS                      = 5
    DXGI_FORMAT_R32G32B32_FLOAT                         = 6
    DXGI_FORMAT_R32G32B32_UINT                          = 7
    DXGI_FORMAT_R32G32B32_SINT                          = 8
    DXGI_FORMAT_R16G16B16A16_TYPELESS                   = 9
    DXGI_FORMAT_R16G16B16A16_FLOAT                      = 10
    DXGI_FORMAT_R16G16B16A16_UNORM                      = 11
    DXGI_FORMAT_R16G16B16A16_UINT                       = 12
    DXGI_FORMAT_R16G16B16A16_SNORM                      = 13
    DXGI_FORMAT_R16G16B16A16_SINT                       = 14
    DXGI_FORMAT_R32G32_TYPELESS                         = 15
    DXGI_FORMAT_R32G32_FLOAT                            = 16
    DXGI_FORMAT_R32G32_UINT                             = 17
    DXGI_FORMAT_R32G32_SINT                             = 18
    DXGI_FORMAT_R32G8X24_TYPELESS                       = 19
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT                    = 20
    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS                = 21
    DXGI_FORMAT_X32_TYPELESS_G8X24_UINT                 = 22
    DXGI_FORMAT_R10G10B10A2_TYPELESS                    = 23
    DXGI_FORMAT_R10G10B10A2_UNORM                       = 24
    DXGI_FORMAT_R10G10B10A2_UINT                        = 25
    DXGI_FORMAT_R11G11B10_FLOAT                         = 26
    DXGI_FORMAT_R8G8B8A8_TYPELESS                       = 27
    DXGI_FORMAT_R8G8B8A8_UNORM                          = 28
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB                     = 29
    DXGI_FORMAT_R8G8B8A8_UINT                           = 30
    DXGI_FORMAT_R8G8B8A8_SNORM                          = 31
    DXGI_FORMAT_R8G8B8A8_SINT                           = 32
    DXGI_FORMAT_R16G16_TYPELESS                         = 33
    DXGI_FORMAT_R16G16_FLOAT                            = 34
    DXGI_FORMAT_R16G16_UNORM                            = 35
    DXGI_FORMAT_R16G16_UINT                             = 36
    DXGI_FORMAT_R16G16_SNORM                            = 37
    DXGI_FORMAT_R16G16_SINT                             = 38
    DXGI_FORMAT_R32_TYPELESS                            = 39
    DXGI_FORMAT_D32_FLOAT                               = 40
    DXGI_FORMAT_R32_FLOAT                               = 41
    DXGI_FORMAT_R32_UINT                                = 42
    DXGI_FORMAT_R32_SINT                                = 43
    DXGI_FORMAT_R24G8_TYPELESS                          = 44
    DXGI_FORMAT_D24_UNORM_S8_UINT                       = 45
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS                   = 46
    DXGI_FORMAT_X24_TYPELESS_G8_UINT                    = 47
    DXGI_FORMAT_R8G8_TYPELESS                           = 48
    DXGI_FORMAT_R8G8_UNORM                              = 49
    DXGI_FORMAT_R8G8_UINT                               = 50
    DXGI_FORMAT_R8G8_SNORM                              = 51
    DXGI_FORMAT_R8G8_SINT                               = 52
    DXGI_FORMAT_R16_TYPELESS                            = 53
    DXGI_FORMAT_R16_FLOAT                               = 54
    DXGI_FORMAT_D16_UNORM                               = 55
    DXGI_FORMAT_R16_UNORM                               = 56
    DXGI_FORMAT_R16_UINT                                = 57
    DXGI_FORMAT_R16_SNORM                               = 58
    DXGI_FORMAT_R16_SINT                                = 59
    DXGI_FORMAT_R8_TYPELESS                             = 60
    DXGI_FORMAT_R8_UNORM                                = 61
    DXGI_FORMAT_R8_UINT                                 = 62
    DXGI_FORMAT_R8_SNORM                                = 63
    DXGI_FORMAT_R8_SINT                                 = 64
    DXGI_FORMAT_A8_UNORM                                = 65
    DXGI_FORMAT_R1_UNORM                                = 66
    DXGI_FORMAT_R9G9B9E5_SHAREDEXP                      = 67
    DXGI_FORMAT_R8G8_B8G8_UNORM                         = 68
    DXGI_FORMAT_G8R8_G8B8_UNORM                         = 69
    DXGI_FORMAT_BC1_TYPELESS                            = 70
    DXGI_FORMAT_BC1_UNORM                               = 71
    DXGI_FORMAT_BC1_UNORM_SRGB                          = 72
    DXGI_FORMAT_BC2_TYPELESS                            = 73
    DXGI_FORMAT_BC2_UNORM                               = 74
    DXGI_FORMAT_BC2_UNORM_SRGB                          = 75
    DXGI_FORMAT_BC3_TYPELESS                            = 76
    DXGI_FORMAT_BC3_UNORM                               = 77
    DXGI_FORMAT_BC3_UNORM_SRGB                          = 78
    DXGI_FORMAT_BC4_TYPELESS                            = 79
    DXGI_FORMAT_BC4_UNORM                               = 80
    DXGI_FORMAT_BC4_SNORM                               = 81
    DXGI_FORMAT_BC5_TYPELESS                            = 82
    DXGI_FORMAT_BC5_UNORM                               = 83
    DXGI_FORMAT_BC5_SNORM                               = 84
    DXGI_FORMAT_B5G6R5_UNORM                            = 85
    DXGI_FORMAT_B5G5R5A1_UNORM                          = 86
    DXGI_FORMAT_B8G8R8A8_UNORM                          = 87
    DXGI_FORMAT_B8G8R8X8_UNORM                          = 88
    DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM              = 89
    DXGI_FORMAT_B8G8R8A8_TYPELESS                       = 90
    DXGI_FORMAT_B8G8R8A8_UNORM_SRGB                     = 91
    DXGI_FORMAT_B8G8R8X8_TYPELESS                       = 92
    DXGI_FORMAT_B8G8R8X8_UNORM_SRGB                     = 93
    DXGI_FORMAT_BC6H_TYPELESS                           = 94
    DXGI_FORMAT_BC6H_UF16                               = 95
    DXGI_FORMAT_BC6H_SF16                               = 96
    DXGI_FORMAT_BC7_TYPELESS                            = 97
    DXGI_FORMAT_BC7_UNORM                               = 98
    DXGI_FORMAT_BC7_UNORM_SRGB                          = 99
    DXGI_FORMAT_AYUV                                    = 100
    DXGI_FORMAT_Y410                                    = 101
    DXGI_FORMAT_Y416                                    = 102
    DXGI_FORMAT_NV12                                    = 103
    DXGI_FORMAT_P010                                    = 104
    DXGI_FORMAT_P016                                    = 105
    DXGI_FORMAT_420_OPAQUE                              = 106
    DXGI_FORMAT_YUY2                                    = 107
    DXGI_FORMAT_Y210                                    = 108
    DXGI_FORMAT_Y216                                    = 109
    DXGI_FORMAT_NV11                                    = 110
    DXGI_FORMAT_AI44                                    = 111
    DXGI_FORMAT_IA44                                    = 112
    DXGI_FORMAT_P8                                      = 113
    DXGI_FORMAT_A8P8                                    = 114
    DXGI_FORMAT_B4G4R4A4_UNORM                          = 115

    DXGI_FORMAT_P208                                    = 130
    DXGI_FORMAT_V208                                    = 131
    DXGI_FORMAT_V408                                    = 132


    DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE         = 189
    DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE = 190

    DXGI_FORMAT_A4B4G4R4_UNORM                          = 191


    DXGI_FORMAT_FORCE_UINT                  = 0xffffffff

    def __str__(self):
        # Return the enum name without the DXGI_FORMAT_ prefix
        return self.name.removeprefix("DXGI_FORMAT_")

    @classmethod
    def _missing_(cls, value):
        """
        Called when DXGI_FORMAT(value) is called with an unknown value.

        If the value is a string matching a substring of the enum name
        (like "R32G32B32A32_TYPELESS"), return the corresponding enum.
        Otherwise, return DXGI_FORMAT_UNKNOWN.
        """
        if isinstance(value, str):
            member_name = f"DXGI_FORMAT_{value}"
            if member_name in cls.__members__:
                return cls.__members__[member_name]
        # Fallback
        return cls.DXGI_FORMAT_UNKNOWN

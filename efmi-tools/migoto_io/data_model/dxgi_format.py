import numpy
import struct

from enum import Enum
from typing import Tuple, Union


class DXGIType(Enum):
    FLOAT32 = (numpy.float32, None, None, None, None)
    FLOAT16 = (numpy.float16, None, None, None, None)
    UINT32 = (numpy.uint32, None, None, None, None)
    UINT16 = (numpy.uint16, None, None, None, None)
    UINT8 = (numpy.uint8, None, None, None, None)
    SINT32 = (numpy.int32, None, None, None, None)
    SINT16 = (numpy.int16, None, None, None, None)
    SINT8 = (numpy.int8, None, None, None, None)
    UNORM16 = (
        numpy.uint16,
        lambda data: numpy.fromiter(data, numpy.float32),
        None,
        lambda data: numpy.around(data * 65535.0).astype(numpy.uint16),
        lambda data: data / 65535.0)
    UNORM8 = (
        numpy.uint8,
        lambda data: numpy.fromiter(data, numpy.float32),
        None,
        lambda data: numpy.around(data * 255.0).astype(numpy.uint8),
        lambda data: data / 255.0)
    SNORM16 = (
        numpy.int16,
        lambda data: numpy.fromiter(data, numpy.float32),
        None,
        lambda data: numpy.around(data * 32767.0).astype(numpy.int16),
        lambda data: data / 32767.0)
    SNORM8 = (
        numpy.int8,
        lambda data: numpy.fromiter(data, numpy.float32),
        None,
        lambda data: numpy.around(data * 127.0).astype(numpy.int8),
        lambda data: data / 127.0)


class DXGIFormat(Enum):
    def __new__(cls, fmt, dxgi_type):
        
        (numpy_type, list_encoder, list_decoder, type_encoder, type_decoder) = dxgi_type.value

        obj = object.__new__(cls)
        obj._value_ = fmt
        obj.format = fmt
        obj.byte_width = 0
        obj.num_values = 0
        obj.value_bit_width = 0
        obj.value_byte_width = 0
        obj.dxgi_type = dxgi_type
        obj.numpy_base_type = numpy_type
        obj.list_encoder = list_encoder
        obj.list_decoder = list_decoder
        obj.type_encoder = type_encoder
        obj.type_decoder = type_decoder

        if list_encoder is None:
            obj.list_encoder = lambda data: numpy.fromiter(data, obj.numpy_base_type)

        if list_decoder is None:
            obj.list_decoder = lambda data: numpy.frombuffer(data, obj.numpy_base_type)

        if type_encoder is None:
            obj.type_encoder = lambda data: data.astype(obj.numpy_base_type)

        if type_encoder is not None:
            obj.encoder = lambda data: obj.type_encoder(obj.list_encoder(data))
        else:
            obj.encoder = obj.list_encoder

        if type_decoder is not None:
            obj.decoder = lambda data: obj.type_decoder(obj.list_decoder(data))
        else:
            obj.decoder = obj.list_decoder

        for value_bit_width, value_byte_width in {'32': 4, '16': 2, '8': 1}.items():
            if value_bit_width in obj.dxgi_type.name:
                obj.num_values = obj.format.count(value_bit_width)
                obj.byte_width = obj.num_values * value_byte_width
                obj.value_bit_width = value_bit_width
                obj.value_byte_width = value_byte_width
                break
    
        if obj.byte_width <= 0:
            raise ValueError(f'Invalid byte width {obj.byte_width} for {obj.format}!')

        return obj

    def get_format(self) -> str:
        return 'DXGI_FORMAT_' + self.format
    
    def get_num_values(self, data_stride = 0) -> int:
        if data_stride > 0:
            # Caller specified data_stride, number of values may differ from the base dtype
            return int(data_stride / self.value_byte_width)
        else:
            return self.num_values

    def get_numpy_type(self, data_stride = 0) -> Union[int, Tuple[Union[numpy.integer, numpy.floating], int]]:
        num_values = self.get_num_values(data_stride)
        # Tuple format of (type, 1) is deprecated, so we have to take special care
        if num_values == 1:
            return self.numpy_base_type
        else:
            return (self.numpy_base_type, num_values)
            
    # Float 32
    R32G32B32A32_FLOAT = 'R32G32B32A32_FLOAT', DXGIType.FLOAT32
    R32G32B32_FLOAT = 'R32G32B32_FLOAT', DXGIType.FLOAT32
    R32G32_FLOAT = 'R32G32_FLOAT', DXGIType.FLOAT32
    R32_FLOAT = 'R32_FLOAT', DXGIType.FLOAT32
    # Float 16
    R16G16B16A16_FLOAT = 'R16G16B16A16_FLOAT', DXGIType.FLOAT16
    R16G16B16_FLOAT = 'R16G16B16_FLOAT', DXGIType.FLOAT16
    R16G16_FLOAT = 'R16G16_FLOAT', DXGIType.FLOAT16
    R16_FLOAT = 'R16_FLOAT', DXGIType.FLOAT16
    # UINT 32
    R32G32B32A32_UINT = 'R32G32B32A32_UINT', DXGIType.UINT32
    R32G32B32_UINT = 'R32G32B32_UINT', DXGIType.UINT32
    R32G32_UINT = 'R32G32_UINT', DXGIType.UINT32
    R32_UINT = 'R32_UINT', DXGIType.UINT32
    # UINT 16
    R16G16B16A16_UINT = 'R16G16B16A16_UINT', DXGIType.UINT16
    R16G16B16_UINT = 'R16G16B16_UINT', DXGIType.UINT16
    R16G16_UINT = 'R16G16_UINT', DXGIType.UINT16
    R16_UINT = 'R16_UINT', DXGIType.UINT16
    # UINT 8
    R8G8B8A8_UINT = 'R8G8B8A8_UINT', DXGIType.UINT8
    R8G8B8_UINT = 'R8G8B8_UINT', DXGIType.UINT8
    R8G8_UINT = 'R8G8_UINT', DXGIType.UINT8
    R8_UINT = 'R8_UINT', DXGIType.UINT8
    # SINT 32
    R32G32B32A32_SINT = 'R32G32B32A32_SINT', DXGIType.SINT32
    R32G32B32_SINT = 'R32G32B32_SINT', DXGIType.SINT32
    R32G32_SINT = 'R32G32_SINT', DXGIType.SINT32
    R32_SINT = 'R32_SINT', DXGIType.SINT32
    # SINT 16
    R16G16B16A16_SINT = 'R16G16B16A16_SINT', DXGIType.SINT16
    R16G16B16_SINT = 'R16G16B16_SINT', DXGIType.SINT16
    R16G16_SINT = 'R16G16_SINT', DXGIType.SINT16
    R16_SINT = 'R16_SINT', DXGIType.SINT16
    # SINT 8
    R8G8B8A8_SINT = 'R8G8B8A8_SINT', DXGIType.SINT8
    R8G8B8_SINT = 'R8G8B8_SINT', DXGIType.SINT8
    R8G8_SINT = 'R8G8_SINT', DXGIType.SINT8
    R8_SINT = 'R8_SINT', DXGIType.SINT8
    # UNORM 16
    R16G16B16A16_UNORM = 'R16G16B16A16_UNORM', DXGIType.UNORM16
    R16G16B16_UNORM = 'R16G16B16_UNORM', DXGIType.UNORM16
    R16G16_UNORM = 'R16G16_UNORM', DXGIType.UNORM16
    R16_UNORM = 'R16_UNORM', DXGIType.UNORM16
    # UNORM 8
    R8G8B8A8_UNORM = 'R8G8B8A8_UNORM', DXGIType.UNORM8
    R8G8B8_UNORM = 'R8G8B8_UNORM', DXGIType.UNORM8
    R8G8_UNORM = 'R8G8_UNORM', DXGIType.UNORM8
    R8_UNORM = 'R8_UNORM', DXGIType.UNORM8
    # SNORM 16
    R16G16B16A16_SNORM = 'R16G16B16A16_SNORM', DXGIType.SNORM16
    R16G16B16_SNORM = 'R16G16B16_SNORM', DXGIType.SNORM16
    R16G16_SNORM = 'R16G16_SNORM', DXGIType.SNORM16
    R16_SNORM = 'R16_SNORM', DXGIType.SNORM16
    # SNORM 8
    R8G8B8A8_SNORM = 'R8G8B8A8_SNORM', DXGIType.SNORM8
    R8G8B8_SNORM = 'R8G8B8_SNORM', DXGIType.SNORM8
    R8G8_SNORM = 'R8G8_SNORM', DXGIType.SNORM8
    R8_SNORM = 'R8_SNORM', DXGIType.SNORM8

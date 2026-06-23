import numpy
import math
import re

from dataclasses import dataclass, field

import bpy

from .data_extractor import BlenderDataExtractor


@dataclass
class ShapeKeyBatch:
    vertex_offset: int = 0
    vertex_count: int = 0
    shapekey_offsets: list = field(default_factory=list)
    vertex_ids: list = field(default_factory=list)
    position_deltas: list = field(default_factory=list)


class ShapeKeyBatchBuilder:
    data_extractor: BlenderDataExtractor

    def __init__(self, data_extractor):
        self.data_extractor = data_extractor

    def get_object_shapekeys(self, obj: bpy.types.Object) -> tuple[dict[int, str], dict[str, numpy.ndarray]]:
        shapekey_pattern = re.compile(r'.*(?:deform|custom)[_ -]*(\d+).*')
        shapekey_ids = {}
        
        for shapekey in obj.data.shape_keys.key_blocks:
            match = shapekey_pattern.findall(shapekey.name.lower())
            if len(match) == 0:
                continue
            shapekey_id = int(match[0])
            shapekey_ids[shapekey_id] = shapekey.name

        if not shapekey_ids:
            return {}, {}

        shapekeys = self.data_extractor.get_shapekey_data(obj, names_filter=list(shapekey_ids.values()), deduct_basis=True)

        return shapekey_ids, shapekeys
    
    def build_shapekey_batches(
        self,
        shapekey_ids: dict[int, str],
        shapekeys: dict[str, numpy.ndarray],
        vertex_ids: numpy.ndarray
    ) -> list[ShapeKeyBatch]:
        
        max_key_id = max(shapekey_ids.keys())
        batch_count = max(1, math.ceil(max_key_id / 126))

        total_vertex_count = 0;

        batches = []
        
        for batch_id in range(batch_count):

            batch = ShapeKeyBatch()

            batch.vertex_offset = total_vertex_count

            # Offsets sequence always starts with 0 for any batch
            batch.shapekey_offsets.append(0)

            shapekeyed_vertex_count = 0
            
            # Single batch contains up to 127 shapekeys (since first value is always 0)
            # So 254 shapekeys should be divided to 2 batches:
            # Batch 0: from 0   to 126 (aka range(0,   127))
            # Batch 1: from 127 to 253 (aka range(127, 254))
            shapekey_id_offset = batch_id * 127

            for shapekey_id in range(shapekey_id_offset, shapekey_id_offset + 127):

                shapekey = shapekeys.get(shapekey_ids.get(shapekey_id, -1), None)
                if shapekey is None or not (-0.00000001 > numpy.min(shapekey) or numpy.max(shapekey) > 0.00000001):
                    batch.shapekey_offsets.append(shapekeyed_vertex_count)
                    continue

                shapekey = shapekey[vertex_ids]

                shapekey_vert_ids = numpy.where(numpy.any(shapekey != 0, axis=1))[0]

                shapekeyed_vertex_count += len(shapekey_vert_ids)
                batch.shapekey_offsets.append(shapekeyed_vertex_count)

                batch.vertex_ids.extend(shapekey_vert_ids)
                batch.position_deltas.extend(shapekey[shapekey_vert_ids])

            total_vertex_count += shapekeyed_vertex_count

            batch.vertex_count = shapekeyed_vertex_count

            batches.append(batch)
    
        return batches
    
    def get_shapekey_batches(self, obj: bpy.types.Object, vertex_ids: numpy.ndarray) -> list[ShapeKeyBatch]:
        if obj.data.shape_keys is None or len(getattr(obj.data.shape_keys, 'key_blocks', [])) == 0:
            print(f'No shapekeys found to process!')
            return []
        
        shapekey_ids, shapekeys = self.get_object_shapekeys(obj)

        if len(shapekeys) == 0:
            return []
        
        batches = self.build_shapekey_batches(shapekey_ids, shapekeys, vertex_ids)

        return batches


@dataclass
class ShapeKeyData:
    batch_configs: numpy.ndarray
    vertex_ids: numpy.ndarray
    position_deltas: numpy.ndarray

    @staticmethod
    def calculate_quantization_scales(position_deltas: numpy.ndarray, batch_count: int) -> list[float]:
        # Per-vertex sums of weighted deltas of all shapekeys for each axis is stored as INT32 (to benefit from InterlockedAdd).
        # Quantization calculations below allow to maximize preserved float precision of each delta while preventing integer overflow.

        # Quantization target defines the maximum safe integer magnitude that a single shapekey batch may contribute to a vertex axis.
        # This way accumulation across all batches cannot overflow INT32 (with abs max of 2147483647).
        # So, based on number of shapekey batches (128 shapekeys each):
        # 1 => 16777215, 2 => 8388607, 3 => 5592405, 4 => 4194303, etc.
        quantization_target = 2147483647 / (batch_count * 128)

        # Since shapekey value is in [0.0, 1.0] range, we can safely allocate the entire quantization target to delta precision.
        # So here we look for the largest absolute delta of each axis and divide quantization target by it.
        # This way, when shader does `weighted_delta * quantization_scale`, result never exceeds quantization_target.
        largest_deltas = [
            numpy.abs(position_deltas[:, 0]).max(),
            numpy.abs(position_deltas[:, 1]).max(),
            numpy.abs(position_deltas[:, 2]).max()
        ]
        quantization_scales, dequantization_scales = [], []
        for delta in largest_deltas:
            quantization_scale = quantization_target / delta if delta != 0 else 0
            quantization_scales.append(quantization_scale)
            dequantization_scales.append(1.0 / quantization_scale if quantization_scale != 0 else 0)

        return quantization_scales, dequantization_scales

    def finalize_batch_configs(self, batches: list[ShapeKeyBatch]):
        # Calculate quantization scales from merged position deltas data.
        quantization_scales, dequantization_scales = self.calculate_quantization_scales(self.position_deltas, len(batches))

        # Store the quantization scales into x, y and z of first configuration entry.
        self.batch_configs[:4] = numpy.array([*quantization_scales, 0.0], dtype=numpy.float32).view(numpy.uint32)

        # Store the dequantization scales into x, y and z of second configuration entry.
        self.batch_configs[4:8] = numpy.array([*dequantization_scales, 0.0], dtype=numpy.float32).view(numpy.uint32)

    @classmethod
    def from_batches(cls, batches: list[ShapeKeyBatch]): 
        batch_configs, vertex_ids, position_deltas = [], [], []

        batch_configs += [0, 0, 0, 0] * 2

        for batch in batches:
            # Each batch config has len of 33, where last entry is [quantization_scales.xyz, batch_vertex_offset] 
            batch_configs += [batch.vertex_offset, 0, 0, 0] + batch.shapekey_offsets
            vertex_ids += batch.vertex_ids
            position_deltas += batch.position_deltas

        data = ShapeKeyData(
            batch_configs   = numpy.array(batch_configs, dtype=numpy.uint32),
            vertex_ids      = numpy.array(vertex_ids, dtype=numpy.uint32),
            position_deltas = numpy.array(position_deltas, dtype=numpy.float16).reshape(-1, 3)
        )

        data.finalize_batch_configs(batches)

        return data

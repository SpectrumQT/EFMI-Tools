import struct
import numpy

# import imageio.v3 as iio

from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

from ...migoto_model.frame_model.resources import Resource

from .migoto_object_builder import MigotoObject, MigotoComponent


@dataclass
class TextureFilter:
    exclude_extensions: list[str]
    exclude_hashes: list[str]
    min_file_size: int

    def is_valid_texture(self, texture: Resource) -> bool:
        if texture.bin_path_deduped is None:
            return False

        # Exclude texture with ignored extension
        if texture.bin_path_deduped.suffix[1:] in self.exclude_extensions:
            return False

        # Exclude textures with specified hashes
        if self.exclude_hashes:
            if texture.hash in self.exclude_hashes:
                return False

        # Exclude texture below minimal file size
        if self.min_file_size != 0:
            file_size = Path(texture.bin_path_deduped).stat().st_size
            if file_size < self.min_file_size:
                return False

        # Exclude non-square textures
        if texture.bin_path_deduped.suffix == '.dds':
            width, height = self.get_dds_dimensions(texture.bin_path_deduped)
            if width != height:
                return False
        elif texture.bin_path_deduped.suffix == '.jpg':
            width, height = self.get_jpg_dimensions(texture.bin_path_deduped)
            if width != height:
                return False

        return True

    @staticmethod
    def get_dds_dimensions(path: Path) -> tuple[int, int]:
        with open(path, 'rb') as f:
            header = f.read(128)

        if header[:4] != b'DDS ':
            raise ValueError('Not a DDS file')

        height, width = struct.unpack_from("<II", header, 12)

        return width, height

    @staticmethod
    def get_jpg_dimensions(path: Path) -> tuple[int, int]:
        with open(path, "rb") as f:
            # JPEG SOI marker
            if f.read(2) != b"\xFF\xD8":
                raise ValueError("Not a JPEG file")

            while True:
                # Find next marker
                byte = f.read(1)
                while byte == b"\xFF":
                    byte = f.read(1)

                if not byte:
                    raise ValueError("Could not find JPEG dimensions")

                marker = byte

                # SOF markers containing dimensions
                if marker in {
                    b"\xC0", b"\xC1", b"\xC2", b"\xC3",
                    b"\xC5", b"\xC6", b"\xC7",
                    b"\xC9", b"\xCA", b"\xCB",
                    b"\xCD", b"\xCE", b"\xCF",
                }:
                    f.read(2)  # block length (unused)
                    f.read(1)  # precision (unused)
                    height, width = struct.unpack(">HH", f.read(4))
                    return width, height

                # Skip segment
                block_length = struct.unpack(">H", f.read(2))[0]
                f.seek(block_length - 2, 1)


@dataclass
class TexturesDescriptor:
    textures: dict[str, Resource] = field(default_factory=dict)
    components_usage: dict[str, list[int]] = field(default_factory=dict)
    slot_usage: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_migoto_object(cls, migoto_object: MigotoObject, texture_filter: TextureFilter):

        descriptor = cls()

        components_usage = defaultdict(list)
        slot_usage = defaultdict(lambda: defaultdict(list))

        for component_id, component in enumerate(migoto_object.components):

            component_name = f"Component {component_id}"

            for slot, textures in component.textures.items():
                
                slot_str = slot.__str__()

                for texture in textures:

                    if not texture_filter.is_valid_texture(texture):
                        continue
                    
                    # Dedupe textures by hash.
                    cached_texture = descriptor.textures.get(texture.hash, None)
                    if cached_texture is not None:
                        if cached_texture.bin_path_deduped != texture.bin_path_deduped:
                            print(f"Skipped texture {texture.hash} with deduped path mismatch (hash collision?): {cached_texture.bin_path_deduped} != {texture.bin_path_deduped}")
                            continue

                    descriptor.textures[texture.hash] = texture

                    # Track which components use this texture.
                    components_usage[texture.hash].append(component_id)

                    # Build usage tokens.
                    tokens = [texture.hash]

                    for shader_type, shader_hash in texture.usage_descriptor.shaders.items():
                        tokens.append(f"{shader_type.value}={shader_hash}")

                    tokens.append(f"{texture.usage_descriptor.call_id:06d}")

                    if texture.data_descriptor.data_format:
                        tokens.append(f"{texture.data_descriptor.data_format}")

                    usage_entry = "-".join(tokens)

                    # Track slot usage.
                    slot_usage[component_name][slot_str].append(usage_entry)

        # Convert defaultdicts back to normal dicts.
        descriptor.components_usage = dict(components_usage)
        descriptor.slot_usage = {component_name: dict(slots) for component_name, slots in slot_usage.items()}

        return descriptor
    
    # def label_textures(self, migoto_object: MigotoObject):

    #     texture_labels = defaultdict(lambda: defaultdict(list))

    #     def get_textures(migoto_component: MigotoComponent, filter_func, sort_by_size: bool = False):
    #         filtered_textures: list[Resource] = []
    #         for slot, textures in migoto_component.textures.items():
    #             for texture in textures:
    #                 if filter_func(slot, texture):
    #                     filtered_textures.append(texture)
    #         if sort_by_size:
    #             filtered_textures.sort(
    #                 key=lambda texture: texture.bin_path_deduped.stat().st_size,
    #                 reverse=True,
    #             )
    #         return filtered_textures
        
    #     def get_texture_slots(texture_hash: str, migoto_component: MigotoComponent):
    #         slots = set()
    #         for slot, textures in migoto_component.textures.items():
    #             for texture in textures:
    #                 if texture_hash == texture.hash:
    #                     slots.add(slot)
    #         return slots

    #     for component_id, component in enumerate(migoto_object.components):

    #         slot_0_textures = get_textures(component, lambda slot, texture: slot.__str__() == "ps-t0", sort_by_size=True)
    #         srgb_textures = get_textures(component, lambda slot, texture: "SRGB" in str(texture.data_descriptor.data_format), sort_by_size=True)
    #         bc5_textures = get_textures(component, lambda slot, texture: "BC5" in str(texture.data_descriptor.data_format), sort_by_size=True)
    #         bc7_textures = get_textures(component, lambda slot, texture: "BC7" in str(texture.data_descriptor.data_format), sort_by_size=True)

    #         if srgb_textures:
    #             if len(srgb_textures) == 1:
    #                 # Select the only SRGB texture.
    #                 texture_labels[srgb_textures[0].hash] = "DIFFUSE"
    #             else:
    #                 srgb_same_size_textures = []
    #                 target_size = srgb_textures[0].bin_path_deduped.stat().st_size
    #                 for texture in srgb_textures:
    #                     if texture.bin_path_deduped.stat().st_size == target_size:
    #                         srgb_same_size_textures.append(texture)
    #                     else:
    #                         break
    #                 if len(srgb_same_size_textures) == 1:
    #                     # Select the largest texture.
    #                     texture_labels[srgb_same_size_textures[0].hash] = "DIFFUSE"
    #                 else:
    #                     # Select the most used texture.
    #                     srgb_same_size_textures.sort(
    #                         key=lambda texture: len(get_texture_slots(texture)),
    #                         reverse=True,
    #                     )
    #                     texture_labels[srgb_same_size_textures[0].hash] = "DIFFUSE"

    #         for texture in self.textures.values():

    #             components_usage = self.components_usage[texture.hash]

    #             if component_id not in components_usage:
    #                 continue
                
    #             # Blender
    #             img = bpy.data.images.load(str(texture.bin_path))
    #             pixels = list(img.pixels)
    #             arr = numpy.array(pixels).reshape(-1, 4)
    #             mean = arr.mean(axis=0)

    #             # # Standalone
    #             # img = iio.imread(str(texture.bin_path))
    #             # mean = img.mean(axis=(0, 1))
                
    #             print(texture.bin_path.name, list(mean * 255))

import shutil
import json

from pathlib import Path
from dataclasses import dataclass

from .textures_descriptor import TexturesDescriptor
from .migoto_object_builder import MigotoObject


@dataclass
class ObjectExporter:

    def export(self, folder_path: Path, migoto_object: MigotoObject, textures_descriptor: TexturesDescriptor) -> None:

        migoto_object.export_to_files(folder_path)

        with open(folder_path / f'TextureUsage.json', "w") as f:
            f.write(json.dumps(textures_descriptor.slot_usage, indent=4))

        for texture_hash, texture in textures_descriptor.textures.items():

            filename = f"Components-{'-'.join(map(str, sorted(list(set(textures_descriptor.components_usage.get(texture_hash))))))}"
            filename += f" t={texture_hash}"
            if texture.data_descriptor.data_format:
                data_format = str(texture.data_descriptor.data_format)
                encoding = data_format.split("_")[0]
                if data_format.endswith("SRGB"):
                    colorspace = "sRGB"
                else:
                    colorspace = "Linear"
                filename += f" {encoding}-{colorspace}"
            filename += texture.bin_path_deduped.suffix

            output_path = folder_path / filename

            shutil.copyfile(texture.bin_path_deduped, output_path)

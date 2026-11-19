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
            if texture.dhash is None:
                print(f"WARNING! Skipped texture without dhash: {texture.bin_path_deduped.name}")
                continue

            components = '-'.join(sorted(map(str, sorted(list(set(textures_descriptor.components_usage.get(texture_hash)))))))
            dxgi_format = texture.data_descriptor.data_format
            extension = texture.bin_path_deduped.suffix

            filename = f"c={components} f={dxgi_format} t={texture_hash} d={texture.dhash}{extension}"

            output_path = folder_path / filename

            shutil.copyfile(texture.bin_path_deduped, output_path)

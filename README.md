<h1 align="center">EFMI Tools</h1>

<h4 align="center">Blender addon for Endfield Model Importer</h4>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#how-to-use">How To Use</a> • 
  <a href="#assets">Assets</a> •
  <a href="#installation">Installation</a> •
  <a href="#resources">Resources</a> •
  <a href="#license">License</a>
</p>

## Disaclaimers

- It is **first public alpha version** of the toolkit that bundles experimental code from early research. Some things may and will not function as expected.

## Known Issues

- Brows of some characters are skipped on extraction/import/export stages. There are some unresolved issues with VB0 data used for them.

## Features

- **Frame Dump Data Extraction** — Fully automatic objects extraction from Endfield frame dumps
- **LOD Data Import** — Fully automatic search for LOD meshes in frame dump and automatic VG matching
- **Extracted Object Import** —Imports extracted object into Blender as fully editable mesh
- **EFMI Mod Export** — Builds plug-and-play EFMI-compatible mod out of mesh components
- **Customizable Export** — Fast template-powered mod export engine

## Planned Features
- **Bones Merging** — Automatic VG lists merging and deduping 

## How To Use

All fields and actions of the plugin have basic tooltips. Refer to [Modder Guide](https://github.com/SpectrumQT/EFMI-TOOLS/blob/main/guides/modder_guide.md) for more details.

## Installation

1. Install [Blender 5.0](https://www.blender.org/download/releases/5-0) or [Blender 4.5 LTS](https://www.blender.org/download/releases/4-5) (not tested with other versions)
2. Download the [latest release](https://github.com/SpectrumQT/EFMI-Tools/releases/latest) of **EFMI-Tools-X.X.X.zip**
3. Open Blender, go to **[Edit] -> [Preferences] -> [Add-ons]**
4. Open addon `.zip` selection dialogue via top-right corner button:
    * Press **[V]** button in the top-right corner and select **Install from Disk...**
5. Locate downloaded **EFMI-Tools-X.X.X.zip** and select it
6. Press **[Install Addon]** button
7. Start typing  **EFMI** to filter in top-right corner
8. Tick checkbox named **Object: EFMI Tools** to enable addon

## Resources

- [XXMI Launcher](https://github.com/SpectrumQT/XXMI-Launcher)
- [EFMI GitHub](https://github.com/SpectrumQT/EFMI) ([Mirror: Gamebanana](https://gamebanana.com/tools/21846))
- [EFMI Tools GitHub (you're here)] ([Mirror: Gamebanana](https://gamebanana.com/tools/21847))
- [Arknights: Endfield Mods - Gamebanana](https://gamebanana.com/games/21842)
- [AGMF Discord Modding Community](https://discord.com/invite/agmf)

## License

EFMI Tools is licensed under the [GPLv3 License](https://github.com/SpectrumQT/EFMI-Tools/blob/main/LICENSE).

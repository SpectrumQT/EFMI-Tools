# EFMI Tools Modder Guide

First of all, please remember that EFMI Tools is in early alpha. If you're not in the mood for pioneering, please consider waiting for it to mature. I haven't even tested it with all characters (let alone all the possible weapons), so bleeding edge algos may easily get confused. Also, it's bundled with known issue of some characters' brows being missing in extracted data (so they can't be modded atm).

This short guide highlights the sequence of actions required to make the most basic mod using the toolkit.

It assumes that you already have basic knowledge about 3dmigoto-based modding, so it won't go too much into the details. To get accustomed with Blender and 3dm modding in general, I highly recommend video guides.

> For the very first mod try to do the basic re-import of original model without any edits (so skip optional steps **#6** and **#8**).

## Mod Creation Steps

1. <a href="#character-menu-dump-creation">Create frame dump in Character Menu.</a>
2. <a href="#objects-extraction-from-character-menu-dump">Extract objects from dump.</a>
3. <a href="#open-world-dump-creation">Create dump in Open World to prepare LoDs data.</a>
4. <a href="#lods-data-import-to-extracted-object">Import LoDs data to extracted object.</a>
5. <a href="#extracted-object-import">Import character object to Blender.</a>
6. <a href="#imported-object-editing">(Optional) Edit components (meshes) in Blender.</a>
    * <a href="#component-naming">Component Naming</a>
    * <a href="#component-structure">Component Structure</a>
    * <a href="#wights">Weights</a>
    * <a href="#modifiers">Modifiers</a>
    * <a href="#shape-keys">Shape Keys</a>
    * <a href="#uv-maps">UV Maps</a>
    * <a href="#vertex-colors">Vertex Colors</a>
    * <a href="#ini-toggles">Ini Toggles</a>
7. <a href="#basic-efmi-mod-export">Export data from Blender to mod folder.</a>
8. <a href="#textures-editing">(Optional) Edit textures in mod folder with DDS editor of your choise.</a>
9. <a href="#view-mod-in-game">View the mod in-game.</a>

## Character Menu Dump Creation

1. Enable dumping via **XXMI Launcher > Settings > EFMI > Enable Hunting**.
2. Make sure that object you're going to dump has no mods applied (idieally temporarily rename `Mods` folder to i.e. `Mods2`).
3. Start the game via **XXMI Launcher**.
4. (Optional) Lower game resolution to 1920x1080 to reduce Frame Dump folder size.
5. Go to Character Menu of character you're going to dump (optionally, go to weapon tab if you also want to grab it).
6. Press **[0]** button on your **NumPad** to enable **Shader Hunting Mod**.
7. Press **[Shift]+[F11]** to create a Frame Dump.
8. Dump folder **FrameAnalysis-DATETIME** will be located in the `EFMI` folder (same folder where `Mods` folder is located).
9. Rename the dump folder to something sensible (i.e. `Endmin Menu Dump` or `Endmin With Weapon Menu Dump`).

## Objects Extraction From Character Menu Dump

1. Go to **Sidebar > Tool > EFMI Tools**.
2. Select Mode: **Extract Objects From Dump**.
3. Configure **Frame Dump** field: input path to **Character Menu** dump folder (i.e. `Endmin Menu Dump`).
4. Configure **Output Folder** filed: input path to folder where you want to store extracted objects (i.e. create `Extracted Objects` folder in `EFMI` folder).
5. Press **[Extract Objects From Dump]** button.
6. Once extraction is complete, folder with extracted objects will contain folders named like `Character 56684` and `Weapon 6416` (where number indicates number of vertices in the model). 
7. Rename the folder(s) to something sensible (i.e. `Endmin Sources` and `Grand Vision Sources`).
8. Open extracted object folder (i.e. `Endmin Sources`) and remove all `.jpg` and `.dds` texture files that you aren't going to mod (or create folder like `Unused Textures` and move them in it for future use). It'll improve in-game performance, reduce resulting mod size and mod.ini clutter.

## Open World Dump Creation

1. Add character to current team.
2. Switch to another character (not the one you're dumping).
    * It will force the game to apply LoD model (currently directly controled character is never getting replaced with LoD).
3. Press **[Shift]+[F11]** to create a Frame Dump.
4. New dump folder **FrameAnalysis-DATETIME** will be located in the `EFMI` folder (same folder where `Mods` folder is located).
5. Rename the dump folder to something sensible (i.e. `Endmin Open World Dump` or `Endmin With Weapon Open World Dump`)

## LoDs Data Import To Extracted Object

1. Go to **Sidebar > Tool > EFMI Tools**.
2. Select Mode: **Import LoDs From Dump**.
3. Configure **Frame Dump** field: input path to **Open World** dump folder (i.e. `Endmin Open World Dump`).
4. Configure **Object Soures** field: input path to extracted object folder generated on step **#2** (i.e. `Endmin Sources`).
4. Press **[Import LoDs From Dump]** button.
5. After ~15 seconds Blender will respond with success message.  
    * You can view details in **Blender Top Menu > Window > Toggle System Console** window.
    * If import results in error instead, try making 1 extra dump. If it errors again, it means there is some issue with dumped objects handling. Please report them.

## Extracted Object Import

1. Go to **Sidebar > Tool > EFMI Tools**
2. Select Mode: **Import Object**
3. Configure **Object Sources** field: input path to extracted object folder generated on step **#2** (i.e. `Endmin Sources`).
4. Press **[Import Object]** 
5. Imported object components will appear as collection of Blender objects named after the object folder.
    * Object will be named same as sources folder name (i.e. `Endmin Sources`).
    * Rename the collection to something more appropriate (i.e. just `Endmin`).
    * Object parts will be named like `Component 0 48e5c5f7` (where `0` is component id and `48e5c5f7` is VB0 hash).
    * Component ids are assigned by toolkit based on the tompost vertex of each component (so horns or hair will always be on top).
    * You can remove the `48e5c5f7` hash or add some text, but **never remove** `Component 0` part (it controls where this Blender object will go to on Mod Export).
 
## Imported Object Editing

### Component Naming

* Component ID determines to which set of shaders each object will go in exported mod.
* Any component object name is valid while it contains 'component' keyword followed by ID (i.e. 'Hat CoMpONEnT- 2 test' is a valid name).
* You can split any Blender object of any component as many Blender objects as you want (or create new ones):
    - Make sure that every Blender object contains 'Component X' (where X is its numerical ID). 
    - In the same way you can create totally new Blender objects for any component (but of course you'll have to fill all the data like weights, UV, etc by yourself).

### Component Structure

* There can be any number of Blender objects with same Component ID inside collection. They will be automatically merged on mod export.
* Every object will have its own drawindexed call inside a call stack with its Component ID.

### Weights

* Please do note that there's no Merged Skeleton feature currently available. It means that Vertex Groups are split between components, so you're limited to whatever set of bones each component originally uses.

### Modifiers

* You can also use **[Apply All Modifiers]** checkbox in mod export option to autoamtically apply exisitng modifiers to temp copies of objects during temp merged object creation on mod export.

### Shape Keys

* Game doesn't use shapekeys, everything is weighted. But feel free to use them in Blender for own convenience!
> I plan to introduce custom shapekeys system to EFMI in the future.

### UV Maps

* Components always contains **TEXCOORD.xy** UV map controlling textures application.

### Vertex Colors

* Components always contains COLOR color attribute which contains some tangent-based autogenerated data used for precise outlines control. If you significantly change the mesh geometry (move or add new verices), please consider painting it black. Algoritm of its generation isn't figured out (yet). If it's not exist during mod export (i.e. for custom mesh), it'll be automatically considered as black.

### Ini Toggles
Toolkit bundles handy GUI system for conditional object display (which can be used to add keybindings to hide/display any Blender objects). 

#### To create the simplest toggle like "Press [9] to hide hat":
1. Expand **Ini Toggles** tab and enable **Use Ini Toggles** checkbox.
2. Press **[Add Var]** button, it will add new block for **TOGGLE_0** toggle var.
3. In **State 1** block press **🧪 Eyedropper Button** icon and select object you want to hide (or click **Object** text field and find by typing name or scrolling).
4. Press **⛮ Gear Button** in the top-right corner of **TOGGLE_0** block, it will open little popup window.
5. Input `9` to **Hotkeys** field and press **[Enter]** (hint: click **[Key Codes]** button to view handy cheat sheet on MicroSoft website with all codes).
6. Next to **TOGGLE_0** name you'll see new text: `[9]`. It means that key was successfully bound.
7. Click on any empty space outside popup window to hide it.

That's it! Once you export your mod, all required ini code will be created automatically. Just press **[9]** in-game, and this object will be hidden. Press **[9]** again, and it'll apear.

> Hint: don't forget that you can split Blender object of any component or creat new ones. This way you can easily create a toggle i.e. for gloves of original in-game model.

## Basic EFMI Mod Export

1. Go to **Sidebar > Tool > EFMI Tools**
2. Select Mode: **Export Mod**
3. Configure **Components** field: Select collection with objects for desired components (i.e. `Endmin`). Skipping arbitrary components is supported, just remove relevant objects from collection. They won't appear in the game.
4. Configure **Object Sources** field: input path to object sources folder (i.e. `Endmin Sources`) containing object data.
5. Configure **Mod Folder** field: input path where you want the exported mod data to be located (i.e. create `Endmin Mod` folder in `Mods` folder).
6. Configure optional mod info fields.
7. Press **[Export Mod]**

## Advanced EFMI Mod Export Options

1. Apply All Modifiers:
    * Automatically applies all existing modifiers to temporary copies of objects created during export.
    * Shapekeyed objects are also supported.
2. Copy Textures:
    * Copy textures from **Object Sources** folder to mod folder. Automatically skips already existing textures.
3. Comment Ini Code:
    * Adds comments to the code in mod.ini. May be useful if you want to get idea about what's going on.
4. Debug Settings > Remove Temp Object:
    * Uncheck to keep temporary objects built from copies of all objects of all components used for export. Primary usecase is EFMI Tools debugging.

## Textures Editing

1. Open exported mod folder (i.e. `Mods/Endmin Mod`).
2. Edit textures with editor of your choisem like **Paint.net** or **Photoshop** (both require dedicated `.dds` plugin installed)).
3. Save textures.
    * Diffuse textures (colors) must be saved in BC7-sRGB (no mipmaps, no gamma shift) format.
    * NormalMap and other textures must be saved in BC7-Linear (no mipmaps, no gamma shift) format.

## View Mod In Game

1. Make sure that character you've modded isn't visible on screen (go to character menu of another character).
    * This step is optional if modded mesh vertex count is the same or below the one that was during previous EFMI (re)load.
2. Press **[F10]** in-game to reload active mods.
3. Your mod should now be visible in-game!

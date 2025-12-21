# GATE simulation example (Ga-68)

This folder contains an example GATE setup to run PET simulations for **Ga-68**.
It is meant as a *reference/example*: you are expected to adapt paths to your own machine
(GATE macros commonly include absolute paths).

## What it does
- Loads a voxelized phantom (activity + attenuation / μ-map).
- Runs a PET simulation using a predefined scanner + physics settings.
- Uses lookup tables to map voxel values to activity and attenuation ranges.
- Uses a material database for the phantom materials.

## Main files
- `main_PR.mac` : main entry macro that includes the rest.
- `my_physics.mac`: physics configuration suitable for PR / Ga-68.
- `phantom_prange.mac`: loads the voxelized phantom and maps voxel labels/ranges.
- `point_source.mac`: defines the radioactive source (currently Ga-68).
- `GateMaterials_Xemis.db` (and/or `Materials.xml`, `myMaterials.xml`): material definitions.
- `ActivityRangefinal.dat`: mapping voxel values → activity range.
- `AttenuationRangefinal.dat`: mapping voxel values → attenuation (μ) range.

## Changing the isotope (source)
This example is configured for **Ga-68**.
If you want to simulate another isotope/source type, edit:
- `point_source.mac`

Typical changes include:
- the isotope name / ion definition
- the decay settings (if applicable)
- energy window / source spectrum settings (depending on your setup)

## Changing phantom size and resolution
If you change phantom **voxel size (resolution)** or **volume dimensions**, you must update:
- `phantom_prange.mac` (voxelized phantom configuration: size, spacing, file path)
- and ensure the activity/attenuation range tables still match your voxel labeling strategy

In other words: the voxel grid in the `.mac` file must match the phantom files you generate.

## Materials limitation (important)
In this example, we are currently limited to **three materials**:
- Water
- Rib bone
- Lung

You are welcome to add more materials:
- extend the material definitions (database/XML)
- and update the range/label mapping files (`ActivityRangefinal.dat`, `AttenuationRangefinal.dat`)
so new voxel labels map to the new materials consistently.

## Notes
- Absolute paths are normal in GATE examples; users should set paths for their environment.
- Keep the voxel label/material mapping consistent across:
  phantom generation → range tables → material definitions → GATE macros.

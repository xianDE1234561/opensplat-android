# Changelog

All notable changes to OpenSplat Android will be documented in this file.

## Unreleased

- Project oriented Gaussian splats as screen-space ellipses instead of large round spots.
- Show initial and refined previews at the top of the reconstruction page and keep the camera where the user leaves it.
- Default local Brush refinement to 300 steps, with a quick 50-step preset and custom iteration counts.
- Ship a single LocusGS shape, 448×256, instead of offering 256×256 / 448×256 / 448×252
  packages. The square export centre-cropped a 16:9 capture to 56 % of its width, while
  448×256 keeps 98.4 % (measured `shiftX` -10 px vs -280 px on a 1280×720 frame); other
  aspects are served by cropping into it. The profile enum, the model picker, and the
  per-aspect install plumbing are gone with it.
- Replace the bottom navigation with Capture, Reconstruction, Lib, and Viewer.
- Offer PocketGS surface-aligned point-cloud seeding and LocusGS initialization in the same local reconstruction workspace. Both feed Brush refinement.
- Browse reconstruction results in Lib and open saved or local PLY/SPZ models in Viewer. Leave previously installed model directories untouched.
- Shrink the single-ABI release APK with R8 and resource compression while keeping native entry points.

## [1.0.0] - 2026-09-20

### Added
- 3D Gaussian Splatting rendering support via customized Filament shaders.
- Real-time on-device Splat training and point cloud reconstruction using LiteRT and Ceres Solver.
- Augmented Reality (ARCore) integration for placing and viewing 3D Gaussian Splats in real-world environments.
- Interactive editing and manipulation tools (bounding box, point-and-click segmentation, transform gizmo).
- Import and export for PLY and SPZ formats.
- Sample Android application demonstrating viewing, training, and AR placement.
- Modernized build system targeting Android SDK 36, Java 21, and 16 KB page size support.

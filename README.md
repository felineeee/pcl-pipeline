# PCL Pipeline
`ONGOING`.

## Preliminary
This project were meant to be a the perception part of my `NIST 1 Assembly/Disassembly` project. This module captures point cloud data from the sensor and passes it through the processing pipeline. The pipeline then attempts to reconstruct the object using geometric algorithms like RANSAC to populate the MoveIt planning scene. However, this approach reveals severe limitations, as fitting complex or irregular geometry becomes highly impractical. Instead using volumetric/octree to voxel grids, directly passing it to planning scene. 

# This code is a simple example of how to use PyVista to create a 3D volumetric texture and warp it non-uniformly.
# It creates a 30x30x30 volume and fills it with a continuous volumetric texture/scalar.
# It then casts the volume to a StructuredGrid so the vertices can be warped non-uniformly.
# It then generates a 3D Vector field matching the dimensions of the dataset and attaches it to the structured volume data block.
# It then executes the warp_by_vector function to warp the volumetric texture.
# It then plots the unwarped and warped volumetric textures.

import pyvista as pv
import numpy as np

# 1. Create dummy 3D volumetric texture data (e.g., 30x30x30 volume)
dims = (30, 30, 30)
grid = pv.ImageData(dimensions=dims)

# Fill it with a continuous volumetric texture/scalar (e.g., a 3D sphere density field)
center = np.array(dims) / 2.0
x, y, z = np.indices(dims)
distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
grid.point_data["Volumetric Texture"] = distance_from_center.flatten()

# 2. IMPORTANT: Cast to a StructuredGrid so the vertices can be warped non-uniformly
structured_volume = grid.cast_to_structured_grid()

# 3. Generate a 3D Vector field matching the dimensions of the dataset
# Example: Displacing points outward from the central axis (exploding effect)
pts = structured_volume.points
vectors = np.zeros_like(pts)

# Let's add a radial shear warp vector field 
vectors[:, 0] = np.sin(pts[:, 2] * 0.2) * 2.0  # X warp depends on Z height
vectors[:, 1] = np.cos(pts[:, 2] * 0.2) * 2.0  # Y warp depends on Z height
vectors[:, 2] = 0.0                            # Keeping Z stable

# 4. Attach vectors to the structured volume data block
structured_volume.point_data["warp_vectors"] = vectors

# 5. Execute warp_by_vector
# The volumetric scalar fields remain completely attached to the points as they move!
warped_volume = structured_volume.warp_by_vector(vectors="warp_vectors", factor=1.0)

# 6. Plotting the warped volumetric texture
pl = pv.Plotter(shape=(1, 2))

# Subplot 1: Cross-section slices of the unwarped volume
pl.subplot(0, 0)
pl.add_text("Original Volume (Orthogonal Slices)")
pl.add_mesh(structured_volume.slice_orthogonal(), scalars="Volumetric Texture", cmap="viridis")

# Subplot 2: Cross-section slices of the warped volume
pl.subplot(0, 1)
pl.add_text("Warped Volumetric Texture")
pl.add_mesh(warped_volume.slice_orthogonal(), scalars="Volumetric Texture", cmap="viridis")

pl.link_views()
pl.show()


import imgaug as ia
import imageio
import numpy as np

# Load an example image (uint8, 128x128x3).
image = ia.quokka(size=(128, 128), extract="square")

# Create an example mask (bool, 128x128).
# Here, we just randomly place a square on the image.
segmap = np.zeros((128, 128), dtype=bool)
segmap[28:71, 35:85] = True
segmap = ia.SegmentationMapOnImage(segmap, shape=image.shape)

# Draw three columns: (1) original image, (2) original image with mask on top, (3) only mask
cells = [
    image,
    segmap.draw_on_image(image),
    segmap.draw(size=image.shape[:2])
]

# Convert cells to grid image and save.
grid_image = ia.draw_grid(cells, cols=3)
imageio.imwrite("example_segmaps_bool.jpg", grid_image)
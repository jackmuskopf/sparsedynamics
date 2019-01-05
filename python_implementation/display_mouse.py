
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from load_img.baseimage import PETImage, CTImage, normalize


# put filepath to image to open
filename = "mpet3715b_em1_v1.pet.img"
filepath = os.path.join("data", filename)

# load image into memory
my_img = PETImage(filepath=filepath)
my_img.load_image()

# convert image to Pandas DataFrame
# pd.DataFrame({'z': my_img.img_data[:, 0]})

# data is [x,y,z,t] array of float64
data = np.swapaxes(my_img.img_data, 0, 2)


# get_axis is just a map from x,y,z,t to the 0,1,2,3 which numpy uses
x_axis = my_img.get_axis("x")
y_axis = my_img.get_axis("y")
z_axis = my_img.get_axis("z")
time_axis = my_img.get_axis("t")

# we can view the 3D image in time from the x,y, or z axes
my_img.img_data = normalize(my_img.img_data)*10 # multiply makes video easier to see
frame_blocks = my_img.split_on_axis(time_axis)	# get 3D frames

# get 2D frames; can also use .sum(axis=view_axis), but i think max looks better
frames_x = [fb.max(axis=x_axis) for fb in frame_blocks]
frames_y = [fb.max(axis=y_axis) for fb in frame_blocks]
frames_z = [fb.max(axis=z_axis) for fb in frame_blocks]

# view as video
fig = plt.figure()
p1 = fig.add_subplot(2, 2, 1)
p2 = fig.add_subplot(2, 2, 2)
p3 = fig.add_subplot(2, 2, 3)
images_x = [[p1.imshow(frame, cmap="gray", clim=(0, 1), animated=True)] for frame in frames_x]
images_y = [[p2.imshow(frame, cmap="gray", clim=(0, 1), animated=True)] for frame in frames_y]
images_z = [[p3.imshow(frame, cmap="gray", clim=(0, 1), animated=True)] for frame in frames_z]
animation_x = animation.ArtistAnimation(fig, images_x, interval=100, blit=True)
animation_y = animation.ArtistAnimation(fig, images_y, interval=100, blit=True)
animation_z = animation.ArtistAnimation(fig, images_z, interval=100, blit=True)
plt.show()


# the PETImage object uses np.memmap to hold data in "memory"
# this just cleans up tempfiles used in np.memmap
# good to do this before exiting program
my_img.unload_image()

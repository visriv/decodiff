import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# video_array = np.load('../../../experiments/kol_diffore_3007/checkpoints/output_seq.npy')
hdf_file = h5py.File('data/kol/results_1.h5', 'r')
video_array = hdf_file['velocity_field'][:]


print(video_array.shape)



frames = [] # for storing the generated images
fig = plt.figure()
for i in range(6000):
    # frames.append([plt.imshow(video_array[0][i][:,:,0], cmap='RdBu_r',animated=True)])
    frames.append([plt.imshow(video_array[i][:,:,0], cmap='RdBu_r',animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=5, blit=True,
                                repeat_delay=1000)
ani.save('movies1.mp4')
plt.show()



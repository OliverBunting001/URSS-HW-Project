import libertem.api as lt
from libertem.udf.raw import PickUDF
import numpy as np
import matplotlib.pyplot as plt
import time

# %% load data
pathname = r'/home/ug-ml/Downloads/data1/default.hdr'
ctx = lt.Context.make_with('inline')

# ctx = lt.Context()

ds = ctx.load('auto', path=pathname)


# %% #zncc image
cc_image = np.zeros(ds.shape.nav)
# roi = frame selector
roi = np.zeros(ds.shape.nav, dtype=bool)

# index of reference frame
roi[0, 0] = True
# reference frame wrapper
d = (ctx.run_udf(dataset=ds, udf=PickUDF(), roi=roi, plots=False))
# reset roi
roi[0, 0] = False
# reference frame data
f_ref = np.squeeze(d['intensity'].data)
# normalised: subtract mean, divide by standard deviation
f_ref_n = (f_ref-np.mean(f_ref))/np.std(f_ref)
# number of pixels
npix = f_ref.shape[0]*f_ref.shape[1]
toc = time.perf_counter()


# compare the reference image with all other images
for i in range(ds.shape[0]):
     for j in range(ds.shape[1]):
         # get frame to compare
         roi[i, j] = True
         d = (ctx.run_udf(dataset=ds, udf=PickUDF(), roi=roi, plots=False))
         roi[i, j] = False
         f_ij = np.squeeze(d['intensity'].data)
         # normalise it
         f_ij_n = (f_ij-np.mean(f_ij))/np.std(f_ij)
         # zero-mean normalised cross-correlation
         zncc = np.sum(f_ref_n*f_ij_n)/npix
         cc_image[i, j] = zncc
tic = time.perf_counter()
duration = tic - toc
print("Calculation took: " + str(duration) + " seconds")

# %% plot
plt.figure()
plt.imshow(cc_image)
plt.axis('off')
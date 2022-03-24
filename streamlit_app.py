import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import SimpleITK as sitk

st.title('Computer Tomography Non-Small Cell Lung Cancer area calculation')
st.markdown("***")
st.markdown('This demo app allows for visualization of a CT scan and appropriate tumor segmentation in three '
            'projections: Axial; Sagittal; Coronal. It shows the tumor contour on top of the CT scan and calculates '
            'the tumor area in mmˆ2')
option = st.selectbox(
     'Choose the scan projection: ',
     ('Axial', 'Sagittal', 'Coronal'))


image = sitk.ReadImage('./data/pat1/image.nrrd')
img_array = sitk.GetArrayFromImage(image)
mask_array = sitk.GetArrayFromImage(sitk.ReadImage('./data/pat1/mask.nrrd'))
img_spacing = image.GetSpacing()

arr = img_array.copy()
msk = mask_array.copy()

fig, ax = plt.subplots()

if option == 'Axial':
     arr = img_array.copy()
     msk = mask_array.copy()
     x = st.slider("Position", 0, len(arr), int(len(arr) / 2))
     img_slice_r = arr[x, ...]
     msk_slice_r = msk[x, ...]
     st.write('NSCLC area: %s [mmˆ2]' % np.round(
          img_spacing[0] * img_spacing[1] * np.round(np.sum(msk[x, ...].flatten()), 2), 2))

elif option == 'Sagittal':
     x = st.slider("Position", 0, arr.shape[1], int(arr.shape[1] / 2))
     arr = img_array.copy()
     msk = mask_array.copy()
     img_slice = np.rot90(arr[..., x].T)
     msk_slice = np.rot90(msk[..., x].T)
     new_shape = (int(img_spacing[0]*img_slice.shape[1]), int(img_spacing[2]*img_slice.shape[0]))
     img_slice_r = cv2.resize(img_slice, new_shape, interpolation=cv2.INTER_NEAREST)
     msk_slice_r = cv2.resize(msk_slice, new_shape, interpolation=cv2.INTER_BITS)
     st.write('NSCLC area: %s [mmˆ2]' % np.round(
          img_spacing[0] * img_spacing[2] * np.round(np.sum(msk_slice_r.flatten()), 2), 2))

elif option == 'Coronal':
     x = st.slider("Position", 0, arr.shape[1], int(arr.shape[1] / 2))
     arr = img_array.copy()
     msk = mask_array.copy()
     img_slice = np.rot90(arr[:, x, :].T)
     msk_slice = np.rot90(msk[:, x, :].T)
     new_shape = (int(img_spacing[0]*img_slice.shape[1]), int(img_spacing[2]*img_slice.shape[0]))
     img_slice_r = cv2.resize(img_slice, new_shape, interpolation=cv2.INTER_NEAREST)
     msk_slice_r = cv2.resize(msk_slice, new_shape, interpolation=cv2.INTER_BITS)
     st.write('NSCLC area: %s [mmˆ2]' % np.round(
          img_spacing[0] * img_spacing[2] * np.round(np.sum(msk_slice_r.flatten()), 2), 2))

ax.imshow(img_slice_r, cmap='bone')
if np.sum(msk_slice_r.flatten()) > 0:
     ax.contour(msk_slice_r, colors='r')

ax.axis('off')
fig.tight_layout()
st.pyplot(fig)
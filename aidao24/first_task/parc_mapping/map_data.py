import nibabel as nib
import numpy as np
from nilearn import datasets

brainnetome_file = './parc_mapping/BN_Atlas_246_1mm.nii'
#schaefer_file = './parc_mapping/Schaefer2018_200Parcels_Kong2022_17Networks_order_FSLMNI152_1mm.nii'
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)

brainnetome_img = nib.load(brainnetome_file)
schaefer_img = nib.load(schaefer_atlas['maps'])

brainnetome_data = brainnetome_img.get_fdata()
schaefer_data = schaefer_img.get_fdata()

brainnetome_regions = np.unique(brainnetome_data)
schaefer_regions = np.unique(schaefer_data)

region_mapping = {}

for schaefer_region in schaefer_regions:
    if schaefer_region == 0:
        continue
    corresponding_voxels = brainnetome_data[schaefer_data == schaefer_region]

    unique, counts = np.unique(corresponding_voxels, return_counts=True)
    brainnetome_region = {}
    for i in range(len(unique)):
        brainnetome_region[unique[i]] = counts[i] / counts.sum()
    
    region_mapping[schaefer_region] = brainnetome_region

print("Schaefer200 to Brainnetome region mapping:")
for schaefer_region, brainnetome_region in region_mapping.items():
    print(f"Schaefer region {schaefer_region} -> Brainnetome region {brainnetome_region}")

data = np.load("ihb.npy")


schaefer200_mask = np.isnan(data).any(axis=(1,2))
brainnetome_mask = ~schaefer200_mask

brainnetome = data[brainnetome_mask]
schaefer200 = data[schaefer200_mask]
#np.nan_to_num(schaefer200, 0)
schaefer200 = schaefer200[~np.isnan(schaefer200)].reshape(160,10,200)

brainnetome_translated_to_schaefer = np.zeros_like(schaefer200)

for n in range(schaefer200.shape[0]):
    for t in range(schaefer200.shape[1]):
        for p in range(schaefer200.shape[2]):
            mapped_regions = region_mapping[p+1]
            for brainnetome_region, weight in mapped_regions.items():
                brainnetome_value = brainnetome[n, t, int(brainnetome_region) - 1]
                brainnetome_translated_to_schaefer[n, t, p] += brainnetome_value * weight
          
processed_data = np.empty((data.shape[0], 10, 200))

processed_data[brainnetome_mask] = brainnetome_translated_to_schaefer
processed_data[schaefer200_mask] = schaefer200          
      
np.save("schaefer200_data.npy", processed_data)

"""
Module: data_loader
===================

This module provides PyTorch Dataset classes for loading and processing medical imaging data stored in NIfTI format, specifically for use in deep learning workflows such as segmentation tasks.

Classes
-------

NiftDataset(Dataset)
    A PyTorch Dataset for loading 2D slices from NIfTI files for multiple patients. Each slice is normalized and returned as a tensor along with its corresponding binary segmentation mask.

    Args:
        patients (List[Patient]): List of Patient objects containing filepaths and metadata.
        filenames (List[str]): List of input modality filenames (e.g., ['CT.nii.gz', 'PET.nii.gz']).
        slice_axis (int, optional): Axis along which to slice the volumes (default: 2, i.e., z-axis).
        verbose (bool, optional): If True, prints additional information and errors.

    Attributes:
        patients (List[Patient]): Patient objects.
        filename_keys (List[str]): Cleaned list of input modality keys.
        slice_axis (int): Axis for slicing.
        verbose (bool): Verbosity flag.
        lesion_label (int): Label value for lesion segmentation (default: 1).
        slice_map (List[Tuple[str, str]]): Mapping from global slice index to (patient_idx, slice_idx_in_patient).
        total_slices (int): Total number of slices across all patients.

    Methods:
        __len__(): Returns the total number of slices.
        __getitem__(global_slice_idx): Returns input and target tensors for the specified slice index.

Nifti3DPatchDataset(Dataset) [Commented Out]
    A PyTorch Dataset for loading 3D patches from NIfTI files for multiple patients. Each patch is normalized and returned as a tensor along with its corresponding binary segmentation mask.

    Args:
        patients (List[Patient]): List of Patient objects.
        filenames_keys (List[str]): List of input modality filenames.
        patch_size (Tuple[int, int, int]): Size of the 3D patch (Depth, Height, Width).
        stride (Optional[Tuple[int, int, int]], optional): Stride for patch extraction (default: patch_size).
        lesion_label (int, optional): Label value for lesion segmentation (default: 1).
        verbose (bool, optional): If True, prints additional information and errors.
        preload_data (bool, optional): If True, preloads all volumes into RAM.

    Attributes:
        patients (List[Patient]): Patient objects.
        filename_keys (List[str]): Cleaned list of input modality keys.
        patch_size (Tuple[int, int, int]): Patch size.
        lesion_label (int): Label value for lesion segmentation.
        verbose (bool): Verbosity flag.
        preload_data (bool): Preload flag.
        stride (Tuple[int, int, int]): Stride for patch extraction.
        patch_map (List[Tuple[int, int, int, int]]): Mapping from global patch index to (patient_idx, d_start, h_start, w_start).
        loaded_volumes (dict): Preloaded volumes if enabled.
        total_patches (int): Total number of patches.

    Methods:
        __len__(): Returns the total number of patches.
        _get_volume_data(patient_idx, key): Returns the volume data for a given patient and modality.
        __getitem__(global_patch_idx): Returns input and target tensors for the specified patch index.

Notes
-----
- Both datasets handle missing files and errors gracefully, returning dummy tensors if necessary.
- Normalization functions (`normalise_slice`, `normalise_volume`) are expected to be defined in `src.utils`.
- The `Patient` class is expected to provide methods for retrieving filepaths and checking file existence.

"""
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from tqdm import tqdm
from src.utils import normalise_slice, normalise_volume
from src.dataset.data_classes import Patient


class NiftDataset(Dataset):
    def __init__(
        self,
        patients: List[Patient],
        filenames: List[str],
        slice_axis=2,  # 0: x, 1: y, 2: z axis to slice the images in the files with.
        verbose=False,
    ):
        super().__init__()

        self.patients = patients
        self.filename_keys = [key.strip() for key in filenames]
        self.slice_axis = int(slice_axis)
        self.verbose = verbose
        self.lesion_label = 1
        self.slice_map: List[Tuple[str, str]] = []

        skipped_dirs = 0
        for patient_idx, patient_obj in tqdm(
            enumerate(self.patients), desc="Loading NIfTI files ", unit="patients"
        ):
            try:
                seg_filepath = patient_obj.seg_filepath
                img_header = nib.load(seg_filepath).header
                n_slices = img_header.get_data_shape()[self.slice_axis]
                for slice_idx_in_patient in range(n_slices):
                    self.slice_map.append((patient_idx, slice_idx_in_patient))
            except Exception as e:
                if self.verbose:
                    print(
                        f"Error loading header for patient {patient_obj.id} (SEG: {seg_filepath}): {e}. Skipping."
                    )
                skipped_dirs += 1

        self.total_slices = len(self.slice_map)
        if self.verbose:
            print(f"Total slices across all patients: {self.total_slices}")
            if skipped_dirs > 0:
                print(f"Skipped {skipped_dirs} directories.")

    def __len__(self) -> int:
        """Returns the total number of slices across all patients."""
        return self.total_slices

    def __getitem__(self, global_slice_idx: int) -> Tuple[torch.tensor, torch.Tensor]:
        """Returns the input and target tensors for a given slice index."""
        list_idx, slice_index = self.slice_map[global_slice_idx]
        current_patient: Patient = self.patients[list_idx]

        input_slices = []
        try:
            for key in self.filename_keys:
                file_path = current_patient.get_input_filepath(key)
                volume = nib.load(file_path).get_fdata()
                img_slice = np.take(volume, indices=slice_index, axis=self.slice_axis)
                img_slice_normalized = normalise_slice(img_slice)
                input_slices.append(img_slice_normalized)

            input_array = np.stack(input_slices, axis=0)

            # Load target segmentation slice from the Patient object
            target_volume = nib.load(current_patient.seg_filepath).get_fdata()
            target_slice = np.take(
                target_volume, indices=slice_index, axis=self.slice_axis
            )

            # Binary mask for the target
            binary_target_array = (target_slice == self.lesion_label).astype(np.float32)
            binary_target_array = np.expand_dims(
                binary_target_array, axis=0
            )  # add channel dimension: (H, W) -> (1, H, W)

            input_tensor = torch.from_numpy(input_array).float()
            target_tensor = torch.from_numpy(binary_target_array).float()

            return input_tensor, target_tensor
        except FileNotFoundError as e:
            print(
                f"FATAL Error (FileNotFound) for global_slice_idx {global_slice_idx} (Patient ID: {current_patient.id}, slice_in_patient: {slice_index}): {e}"
            )
            dummy_h, dummy_w = 256, 256
            return torch.zeros(
                (len(self.input_filenames_keys), dummy_h, dummy_w), dtype=torch.float32
            ), torch.zeros((1, dummy_h, dummy_w), dtype=torch.float32)
        except Exception as e:
            print(
                f"FATAL Error during __getitem__ for global_slice_idx {global_slice_idx} (Patient ID: {current_patient.id}, slice_in_patient: {slice_index}): {e}"
            )
            dummy_h, dummy_w = 256, 256
            return torch.zeros(
                (len(self.input_filenames_keys), dummy_h, dummy_w), dtype=torch.float32
            ), torch.zeros((1, dummy_h, dummy_w), dtype=torch.float32)

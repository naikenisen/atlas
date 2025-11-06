import nibabel as nib
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Patient:
    """Class for keeping track of a patient."""

    id: str
    ctres_filepath: str
    ct_filepath: str
    pet_filepath: str
    suv_filepath: str
    seg_filepath: str
    has_segmentation: Optional[bool] = None

    def get_input_filepath(self, filename_key: str) -> str:
        """Helper to get specific input modality filepath based on a key."""
        key = filename_key.upper()
        if key == "PET.NII.GZ" or key == "PET.NII":
            return self.pet_filepath
        elif key == "CT.NII.GZ" or key == "CT.NII":
            return self.ct_filepath
        elif key == "CTRES.NII.GZ" or key == "CTRES.NII":
            return self.ctres_filepath
        elif key == "SEG.NII.GZ" or key == "SEG.NII":
            return self.seg_filepath
        elif key == "SUV.NII.GZ" or key == "SUV.NII":
            return self.suv_filepath
        else:
            raise ValueError(f"Unknown input filename key: {filename_key}")

    @staticmethod
    def from_folder_path(folder_path: str) -> "Patient":
        """
        Create a Patient object from a folder_path.

        Args:
            folder_path (str): folder_path containing the subject data.
        """
        # find the files in the folder_path recursively (they can be in subfolders)
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        # filter the files to find the ones we need
        ctres_filepath = [f for f in files if "CTres.nii.gz" or "CTres.nii" in f][0]
        ct_filepath = [f for f in files if "CT.nii.gz" or "CT.nii" in f][0]
        pet_filepath = [f for f in files if "PET.nii.gz" or "PET.nii" in f][0]
        suv_filepath = [f for f in files if "SEG.nii.gz" or "SEG.nii" in f][0]
        seg_filepath = [f for f in files if "SEG.nii.gz" or "SEG.nii" in f][0]

        # check if the files exist
        if not os.path.exists(ctres_filepath):
            raise FileNotFoundError(f"File {ctres_filepath} does not exist")
        if not os.path.exists(ct_filepath):
            raise FileNotFoundError(f"File {ct_filepath} does not exist")
        if not os.path.exists(pet_filepath):
            raise FileNotFoundError(f"File {pet_filepath} does not exist")
        if not os.path.exists(suv_filepath):
            raise FileNotFoundError(f"File {suv_filepath} does not exist")
        if not os.path.exists(seg_filepath):
            raise FileNotFoundError(f"File {seg_filepath} does not exist")

        # check if the segmentation file has non zero values
        seg_img = nib.load(seg_filepath)
        seg_data = seg_img.get_fdata()
        has_segmentation = True if seg_data.any() else False

        return Patient(
            id=os.path.basename(folder_path),
            ctres_filepath=ctres_filepath,
            ct_filepath=ct_filepath,
            pet_filepath=pet_filepath,
            suv_filepath=suv_filepath,
            seg_filepath=seg_filepath,
            has_segmentation=has_segmentation,
        )

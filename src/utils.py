import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import os
from tqdm import tqdm
from src.dataset.data_classes import Patient
from typing import Optional, Tuple


def find_file(directory: str, filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Recursively searches for a file within a directory and its subdirectories.

    Args:
        directory (str): The root directory to start the search.
        filename (str): The name of the file to find.

    Returns:
        tuple[Optional[str], Optional[str]]: A tuple containing the directory path and
                    the full path to the file if found. If not found, returns (None, None).
                    The first element is the directory path, and the second element is
                    the full path to the file.
                    If the file is not found, both elements will be None.
    """
    for root, _, files in os.walk(directory):
        if filename in files:
            return root.strip(), os.path.join(root, filename).strip()
    return None, None


def get_patients(data_folder_path: str = "data/") -> list:
    """
    Get a list of patient directories from the given data folder path.
    Args:
        data_folder_path (str): Path to the folder containing patient directories.
    Returns:
        list: List of patient directory names.
    """
    try:
        patients = [
            Patient.from_folder_path(os.path.join(data_folder_path, d))
            for d in os.listdir(data_folder_path)
            if os.path.isdir(os.path.join(data_folder_path, d))
        ]
        return patients
    except FileNotFoundError:
        print(
            f"Error: Directory not found at {data_folder_path}. Please ensure the path is correct."
        )
        return []


def get_scan_data(file_path: str) -> tuple:
    """
    Load a NIfTI scan file and return the data and header information.
    Args:
        file_path (str): Path to the NIfTI file.
    Returns:
        tuple: A tuple containing the scan header and the scan data as a NumPy array.
    """
    try:
        scan_img = nib.load(file_path)
        scan_header = scan_img.header
        scan_data = scan_img.get_fdata()

        return (scan_header, scan_data)

    except FileNotFoundError:
        print(
            f"Error: File not found at {file_path}. Please ensure the path is correct."
        )
        print("Please update 'data_folder_path' in the script.")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def visualize_segmentation_with_highlights(
    patient: Patient,
    slice_index: int = None,
    title: str = "Segmentation Visualization",
    slice_axis: int = 2,
):
    """
    Visualizes the segmentation volume by showing slices along a specified axis.
    Highlights the contours of the segmented regions.

    Args:
    - patient (Patient): The patient object containing the segmentation volume.
    - slice_axis (int): The axis along which to slice the volume (0: x, 1: y, 2: z).
    """
    seg_volume = nib.load(patient.seg_filepath).get_fdata()
    num_slices = seg_volume.shape[slice_axis]
    segmentation_found_overall = False

    print(f"Scanning {num_slices} slices along axis {slice_axis}...")
    if slice_index is not None:
        print(f"Visualizing slice {slice_index} (axis {slice_axis}) only.")
        current_slice = np.take(seg_volume, indices=slice_index, axis=slice_axis)
        if np.any(current_slice > 0):
            segmentation_found_overall = True
            print(f"Segmentation found on slice {slice_index} (axis {slice_axis}).")

            plt.figure(figsize=(8, 8))
            plt.imshow(
                current_slice,
                cmap="gray",
                vmin=np.min(seg_volume),
                vmax=np.max(seg_volume),
            )
            contours = find_contours(current_slice, level=0)

            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color="red")
            plt.title(title)
            plt.axis("off")
            plt.show()
            return
    else:
        for i in tqdm(range(num_slices)):
            current_slice = np.take(seg_volume, indices=i, axis=slice_axis)
            if np.any(current_slice > 0):
                segmentation_found_overall = True
                print(f"Segmentation found on slice {i} (axis {slice_axis}).")

                plt.figure(figsize=(8, 8))
                plt.imshow(
                    current_slice,
                    cmap="gray",
                    vmin=np.min(seg_volume),
                    vmax=np.max(seg_volume),
                )
                contours = find_contours(current_slice, level=0)

                for contour in contours:
                    plt.plot(
                        contour[:, 1], contour[:, 0], linewidth=2, color="red"
                    )  # x, y are swapped for plot

                plt.title(f"{title}: Slice {i} (Segmentation Detected)")
                plt.axis("off")
                plt.show()

                return

    if not segmentation_found_overall:
        print(
            f"No segmentation found in any slice of the volume (using threshold > 0)."
        )


def visualise_scan_data_sample(scan_data: np.ndarray, scan_index: int, title: str):
    # --- Visualize a slice (if data was loaded successfully) ---
    if scan_data is not None:
        slice_index = scan_data.shape[1] // 2 if scan_index is None else scan_index
        scan_slice = scan_data[:, :, slice_index]

        plt.figure(figsize=(8, 8))
        plt.imshow(scan_slice, cmap="gray")  # 'hot' or 'jet' can also be good for PET
        plt.title(title)
        # plt.colorbar()
        plt.axis("off")  # Hide axes
        plt.show()
    else:
        print("Skipping visualization as data could not be loaded.")


def visualise_scan_data_animation(
    scan_data: np.ndarray,
    slice_axis=2,
    title_prefix="Scan Slice",
    cmap="gray",
    interval=100,
):
    """
    Visualize the scan data as an animation.
    Args:
        scan_data (np.ndarray): The scan data to visualize.
    """
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display

    if not isinstance(scan_data, np.ndarray) or scan_data.ndim != 3:
        print("Error: scan_data must be a 3D NumPy array.")
        return None

    num_slices = scan_data.shape[slice_axis]

    # Create figure and axes
    fig, ax = plt.subplots()

    # Function to get the i-th slice based on slice_axis
    def get_slice(frame_index):
        if slice_axis == 0:
            return scan_data[frame_index, :, :]
        elif slice_axis == 1:
            return scan_data[:, frame_index, :]
        elif slice_axis == 2:
            return scan_data[:, :, frame_index]
        else:
            raise ValueError("slice_axis must be 0, 1, or 2")

    # Initialize the plot with the first slice
    im = ax.imshow(get_slice(0), cmap=cmap)
    ax.set_title(f"{title_prefix}: 0 / {num_slices-1}")
    ax.axis("off")

    plt.close(fig)

    # Update function for the animation
    def update(frame_index):
        im.set_data(get_slice(frame_index))
        ax.set_title(f"{title_prefix}: {frame_index} / {num_slices-1}")
        return [im]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_slices, interval=interval, blit=True)

    # Convert the animation to HTML5 video for display in Jupyter
    return anim, HTML(anim.to_html5_video())


def normalise_slice(image_slice: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to the range [0, 1]."""
    min_val = np.min(image_slice)
    max_val = np.max(image_slice)

    return (
        (image_slice - min_val) / (max_val - min_val)
        if max_val - min_val > 1e-6
        else image_slice
    )


def normalise_volume(volume_patch: np.ndarray) -> np.ndarray:
    """Normalizes a 3D patch to the range [0, 1]."""
    patch = volume_patch.astype(np.float32)
    min_val, max_val = np.min(patch), np.max(patch)

    if max_val - min_val > 1e-6:
        return (patch - min_val) / (max_val - min_val)
    else:
        return patch

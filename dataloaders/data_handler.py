import nibabel as nib
import numpy as np
import yaml
import json

def read_yaml_file(file_path):
    """
    Read a YAML file and return its contents as a Python dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed contents of the YAML file.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)  # Use safe_load to avoid executing arbitrary code
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


def save_3d_image_nifti(image, affine, output_path):
    """
    Save a 3D image as a NIfTI file.

    Args:
        image (numpy.ndarray): 3D array representing the image to save.
        affine (numpy.ndarray): 4x4 array defining the affine transformation for the image.
        output_path (str): File path to save the NIfTI file (e.g., 'output_image.nii.gz').

    Returns:
        None
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Input image must be a 3D numpy array.")
    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("Affine must be a 4x4 numpy array.")

    # Create a NIfTI image
    nifti_image = nib.Nifti1Image(image, affine)

    # Save the image to the specified file path
    nib.save(nifti_image, output_path)

    print(f"Image saved successfully to {output_path}")


def load_nifti_image(file_path):
    """
    Load a NIfTI image from a file.

    Args:
        file_path (str): Path to the NIfTI file (e.g., 'image.nii.gz').

    Returns:
        image_data (numpy.ndarray): 3D (or 4D) array containing the image data.
        affine (numpy.ndarray): 4x4 array representing the affine transformation matrix.
        header (nib.Nifti1Header): Header information of the NIfTI file.
    """
    # Load the NIfTI file using nibabel
    nifti_image = nib.load(file_path)

    # Extract the image data, affine, and header
    image_data = nifti_image.get_fdata(dtype=np.float32)  # Load as float32
    # affine = nifti_image.affine
    # header = nifti_image.header

    return image_data


def read_json_file(file_path):
    """
    Reads a JSON file and returns its contents as a dictionary.

    :param file_path: Path to the JSON file.
    :return: Dictionary containing the JSON data.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



def save_json_file(data, file_path, indent=4):
    """
    Saves a Python dictionary or list to a JSON file.

    :param data: The data to save (e.g., dictionary or list).
    :param file_path: Path where the JSON file will be saved.
    :param indent: Indentation level for pretty-printing (default 4).
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=indent)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving to {file_path}: {e}")

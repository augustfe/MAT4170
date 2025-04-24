from pathlib import Path

import numpy as np

src_dir = Path(__file__).parent
data_dir = src_dir.parent / "data"


def retrieve_dataset(file_nr: int) -> np.ndarray:
    """Load data from a file.

    Args:
        file_nr (int): The number corresponding to the file "hj{file_nr}.dat".

    Returns:
        np.ndarray: The loaded data, of shape (x_i, y_i, z_i), for i = 1, ..., n.
    """
    if not isinstance(file_nr, int):
        raise TypeError("file_nr must be an integer")
    if not (1 <= file_nr <= 9):
        raise ValueError("file_nr must be between 1 and 9")

    file = data_dir / f"hj{file_nr}.dat"
    arr = np.loadtxt(file)

    return arr


def chord_length_parametrization(data: np.ndarray) -> np.ndarray:
    """Calculate the chord length parametrization of a 3D curve.

    Args:
        array (np.ndarray): The input array of shape (x_i, y_i, z_i), for i = 1, ..., n.

    Returns:
        np.ndarray: The chord length parametrization of the input array.
    """
    arr_prev = data[:-1]
    arr_next = data[1:]
    diff = arr_next - arr_prev
    dist: np.ndarray = np.sqrt((diff**2).sum(axis=1))

    dist_padded = np.insert(dist, 0, 0)
    u_arr = np.cumsum(dist_padded)

    return u_arr


def add_chord_length(data: np.ndarray) -> np.ndarray:
    """Add the chord length parametrization to the data.

    Args:
        data (np.ndarray): The input array of shape (x_i, y_i, z_i), for i = 1, ..., n.

    Returns:
        np.ndarray: The input array with the chord length parametrization added.
    """
    u_arr = chord_length_parametrization(data)
    return np.column_stack((u_arr, data))


def load(file_nr: int) -> np.ndarray:
    """Load the data and add the chord length parametrization.

    Args:
        file_nr (int): The number corresponding to the file "hj{file_nr}.dat".

    Returns:
        np.ndarray: The loaded data with the chord length parametrization added.
    """
    data = retrieve_dataset(file_nr)
    return add_chord_length(data)


if __name__ == "__main__":
    data = retrieve_dataset(1)
    add_chord_length(data)

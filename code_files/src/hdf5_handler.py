import h5py


def read_hdf5_to_dict(file: str) -> dict:
    """
    Load a hdf5 file and return all data in there as a dictionary.
    The attributes of the dataset are saved as a dictionary too.

    :param file: str to file-location
    :return: dict with all the data and their attributes
    """
    with h5py.File(file, "r") as f:
        lst_keys = list(f.keys())
        data_dict = {}

        # iterate over all datasets in the file
        for key in lst_keys:
            data_hdf = f[key]
            all_attrs = list(data_hdf.attrs.keys())
            parameter_dict = {}

            # iterate over all keys of the current dataset
            for attribute in all_attrs:
                parameter_dict[attribute] = data_hdf.attrs[attribute]

            # write the dataset into the dictionary
            data_dict[key] = data_hdf[()]
            # write the attributes of the dataset to the dictionary as a "dataset_parameters" entry
            data_dict[key + "_parameters"] = parameter_dict

        f.close()

    return data_dict


def to_hdf5_from_dict(file: str, data_dict: dict) -> None:
    """
    Use the data_dict to write a hdf5 file.
    The attributes of each dataset can be passed as dict under 'key'+_parameters.

    :param file: str to file-location
    :param data_dict: dict with all the data and their attributes
    :return:
    """

    with h5py.File(file, "w") as f:
        # iterate over all keys in dict and divide them into data and parameters
        for key, value in data_dict.items():
            if '_parameters' in key:
                pass
            else:
                # process data:
                f.create_dataset(key, data=value, compression="gzip", compression_opts=9)

        for key, value in data_dict.items():
            if '_parameters' in key:
                # process attributes:
                data_hdf = f[key[:-11]]  # solely key without _parameters
                for p_key, p_value in value.items():
                    data_hdf.attrs[p_key] = p_value
            else:
                pass

        f.close()
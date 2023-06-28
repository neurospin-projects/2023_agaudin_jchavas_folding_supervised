import numpy as np
try:
    from soma import aims
except ImportError:
    print("INFO: you are not in a brainvisa environment. Probably OK.")



def build_converter():
    """Build the linear transition between the end of 
    the encoders and the common projection heads."""
    return None

def bv_checks(model, filenames):
    vol_file = f"{model.config.data[0].crop_dir}/{filenames[0]}" +\
               f"{model.config.data[0].crop_file_suffix}"
    vol = aims.read(vol_file)
    model.sample_ref_0 = np.asarray(vol)
    if not np.array_equal(model.sample_ref_0[..., 0],
                            model.sample_k[0, 0, ...]):
        raise ValueError(
            "Images files don't match!!!\n"
            f"Subject name = {filenames[0]}\n"
            f"Shape of reference file = "
            f"{model.sample_ref_0[...,0].shape}\n"
            f"Shape of file read from array = "
            f"{model.sample_k[0,0,...].shape}\n"
            f"Sum of reference file = "
            f"{model.sample_ref_0.sum()}\n"
        f"Sum of file read from array = "
            f"{model.sample_k[0,...].sum()}")
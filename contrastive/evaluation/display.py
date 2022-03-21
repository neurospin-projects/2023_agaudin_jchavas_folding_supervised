""" The aim of this script is to display buckets thanks to Anatomist.
"""

import anatomist.api as anatomist
from soma import aims
import glob

a = anatomist.Anatomist()


def display_bucket(src_dir: str, subject_id: str):
    """Displays the chosen bucket with bucket files in src_dir"""
    global a
    block = a.AWindowsBlock(a, 2)
    bucket_file = [s for s in glob.glob(f"{src_dir}/*.bck")
                   if subject_id in s][0]
    bucket = aims.read(bucket_file)
    a_bucket = a.toAObject(bucket)
    globals()['w3d'] = a.createWindow("3D", block=block)
    globals()['w3d'].addObjects(a_bucket)
    globals()['w3d1'] = a.createWindow("3D", block=block)
    globals()['w3d1'].addObjects(a_bucket)
    return block


def display_several_buckets(src_dir: str, subject_id_list: list, prefix: str):
    global a
    bucket_file_l = []
    for subject_id in subject_id_list:
        bucket_file = [s for s in glob.glob(f"{src_dir}/*.bck")
                       if subject_id in s][0]
        bucket_file_l.append(bucket_file)
    print(bucket_file_l)
    a_bucket_l = []
    for idx, bucket_file in enumerate(bucket_file_l):
        bucket = aims.read(bucket_file)
        a_bucket = a.toAObject(bucket)
        a_bucket.setName(f"{prefix}_{subject_id_list[idx]}")
        a_bucket_l.append(a_bucket)
    block = a.AWindowsBlock(a, 2)
    for idx, a_b in enumerate(a_bucket_l):
        window_name = f"w_{subject_id_list[idx]}"
        globals()[window_name] = a.createWindow("3D",
                                                geometry=[0, 0, 500, 500],
                                                block=block)
        globals()[window_name].addObjects(a_b)
        # globals()[window_name].camera(view_quaternion=\
        #                               [-0.361454546451569, 0.398135215044022,
        #                               0.458537578582764, 0.707518458366394],
        #                               zoom=1.)
    return block


if __name__ == '__main__':
    src_dir = '/neurospin/dico/data/deep_folding/papers/midl2022/crops/'\
              'CINGULATE/mask/sulcus_based/2mm/centered_combined/hcp'
    bucket_dir = f"{src_dir}/Rbuckets"

    # # From represeration space viewpoint
    # block_smallest = display_several_buckets(bucket_dir,
    #     ['644044', '198653',
    #      '894067', '199150',
    #      '111211', '117021',
    #      '176744', '479762'
    #      ],
    #     'smallest')

    # block_biggest = display_several_buckets(bucket_dir,
    #     ['352132', '406836',
    #      '103111', '784565',
    #      '943862', '318637',
    #      '173435', '114924'],
    #     'biggest')

    # Using output distances
    block_smallest = display_several_buckets(bucket_dir,
        ['568963', '107422',
         '318637', '174437',
         '792766', '248339',
         '644044', '198653'
         ],
        'smallest')

    block_biggest = display_several_buckets(bucket_dir,
        ['833148', '406836',
         '150928', '707749',
         '111009', '159845',
         '723141', '212015'],
        'biggest')

    from soma.qt_gui.qt_backend import Qt
    Qt.qApp.exec_()

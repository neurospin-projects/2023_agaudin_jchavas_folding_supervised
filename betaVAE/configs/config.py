from datetime import datetime
now = datetime.now()


class Config:

    def __init__(self):
        self.batch_size = 64
        self.nb_epoch = 50
        self.kl = 2
        self.n = 10
        self.lr = 2e-4
        self.in_shape = (1, 20, 40, 40) # input size with padding
        #self.data_dir = "/path/to/data/directory"
        #self.subject_dir = "/path/to/list_of_subjects"
        #self.save_dir = "/path/to/saving/directory"
        self.save_dir = f"/neurospin/dico/agaudin/Runs/09_new_repo/Output/beta-VAE/{now:%Y-%m-%d}/{now:%H-%M-%S}/"
        self.data_dir = "/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/2mm/CINGULATE/mask"
        self.subject_dir = "/neurospin/dico/data/deep_folding/papers/miccai2023/Input/datasets/hcp/crops/2mm/CINGULATE/mask/Rskeleton_subject_full.csv"
        # used for embeddings generation:
        self.acc_subjects_dir = "/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/crops/2mm/CINGULATE/mask"
        self.test_model_dir = "/neurospin/dico/agaudin/Runs/09_new_repo/Output/2023-04-05/11-53-51"


# class Config:

#     def __init__(self):
#         self.batch_size = 64
#         self.nb_epoch = 500 #300
#         self.kl = 2
#         self.n = 10
#         self.lr = 2e-4
#         self.in_shape = (1, 20, 40, 40) # input size with padding
#         self.save_dir = f"/neurospin/dico/lguillon/collab_joel_aymeric_cingulate/n_{self.n}/"
#         self.data_dir = "/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/2mm/CINGULATE/mask/"
#         self.subject_dir = "/neurospin/dico/data/deep_folding/papers/midl2022/HCP_half_1bis.csv"
#         self.acc_subjects_dir = "/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/crops/2mm/CINGULATE/mask"
#         self.test_model_dir = f"/neurospin/dico/lguillon/collab_joel_aymeric_cingulate/n_{self.n}/#5"

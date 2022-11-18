
class Config:

    def __init__(self):
        self.batch_size = 64
        self.nb_epoch = 500 #300
        self.kl = 2
        self.n = 10
        self.lr = 2e-4
        self.in_shape = (1, 20, 40, 40) # input size with padding
        self.save_dir = f"/neurospin/dico/lguillon/collab_joel_aymeric_cingulate/n_{self.n}/"
        self.data_dir = "/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/2mm/CINGULATE/mask/"
        self.subject_dir = "/neurospin/dico/data/deep_folding/papers/midl2022/HCP_half_1bis.csv"
        self.acc_subjects_dir = "/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/crops/2mm/CINGULATE/mask"
        self.test_model_dir = f"/neurospin/dico/lguillon/collab_joel_aymeric_cingulate/n_{self.n}/#5"

# -*- coding: utf-8 -*-
from numpy.linalg import norm
import time
import numpy as np


class UCFModel(object):
    """
    UserBasedCF
    """
    def __init__(self):
        self.user_loc_score = None

    def save_result(self, path):
        ctime = time.time()   
        print("Saving UCF the result...")
        np.save(path + "user_loc_score_1_20", self.user_loc_score)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_result(self, path):
        ctime = time.time()
        print("Loading UCF result...")
        self.user_loc_score = np.load(path + "user_loc_score_1_20.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def pre_compute_user_loc_scores(self, UL):
        ctime = time.time()
        print("Training UCFModel...")
        sim = UL.dot(UL.T)
        second_norms = [norm(UL[i]) for i in range(UL.shape[0])]
        for i in range(UL.shape[0]):
            sim[i][i] = 0.0
            for j in range(i+1, UL.shape[0]):
                if (second_norms[i]!=0.0 and second_norms[j] != 0.0):
                    sim[i][j] /= (second_norms[i] * second_norms[j])
                    sim[j][i] /= (second_norms[i] * second_norms[j])
                else:
                    sim[i][j] = 0.0
                    sim[j][i] = 0.0
        self.user_loc_score = sim.dot(UL)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, uid, lid):
        return self.user_loc_score[uid][lid]

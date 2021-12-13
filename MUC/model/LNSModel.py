# -*- coding: utf-8 -*-
import time
import numpy as np
import random


class LNSModel(object):
    """
    LNSModel
    """

    def __init__(self, lamb):
        self.uid_lid_list = None
        self.lamb = lamb
        self.w_1 = np.array([random.uniform(0.0, 1.0) for i in range(5)])
        self.w_2 = np.array([random.uniform(0.0, 1.0) for i in range(5)])
        self.w_3 = np.array([random.uniform(0.0, 1.0) for i in range(5)])
        self.para = np.array([0.0, 0.0, 0.0, 0.0])

    def save_result(self, path):
        ctime = time.time()
        print("Saving the LNS result...")
        np.save(path + "LNS_1_20", self.para)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_result(self, path, LF, NF, SF):
        ctime = time.time()
        print("Loading LNS result...")
        self.para = np.load(path + "LNS_1_20.npy")
        self.LF = LF
        self.NF = NF
        self.SF = SF
        print(self.para)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def sgd_lns(self, max_iters, uid_lid_list, user_features, LF, NF, SF):
        print("Training LNSModel...")
        self.LF = LF
        self.NF = NF
        self.SF = SF

        def sigmoid(x):
            return 1.0 / (1 + np.exp(-x))

        w_1 = self.w_1
        w_2 = self.w_2
        w_3 = self.w_3
        lamb = self.lamb
        learn_rate = 0.005
        for iters in range(max_iters):
            random.shuffle(uid_lid_list)
            for uid, lid in uid_lid_list:
                phi_1 = sigmoid(w_1.dot(user_features[uid][1:6].T))
                phi_2 = sigmoid(w_2.dot(user_features[uid][6:11].T))
                phi_3 = sigmoid(w_3.dot(user_features[uid][11:16].T))

                d_w1 = 2 * lamb * w_1 - phi_1 * (1 - phi_1) * user_features[uid][1:6] * LF[uid, lid]
                d_w2 = 2 * lamb * w_2 - phi_2 * (1 - phi_2) * user_features[uid][6:11] * NF[uid, lid]
                d_w3 = 2 * lamb * w_3 - phi_3 * (1 - phi_3) * user_features[uid][11:16] * SF[uid, lid]

                w_1 = w_1 - learn_rate * d_w1
                w_2 = w_2 - learn_rate * d_w2
                w_3 = w_3 - learn_rate * d_w3

        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.phi_3 = phi_3
        self.para = np.array([phi_1, phi_2, phi_3])
        print(phi_1, phi_2, phi_3)
        print("Done LNSModel...")
        return phi_1, phi_2, phi_3

    def predict(self, uid, lid):
        phi_1 = self.para[0]
        phi_2 = self.para[1]
        phi_3 = self.para[2]
        return phi_1 * self.LF[uid][lid] + phi_2 * self.NF[uid][lid] + phi_3 * self.SF[uid][lid]

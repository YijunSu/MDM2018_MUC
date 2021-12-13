# -*- coding: utf-8 -*-
from __future__ import division
import time
import numpy as np
from collections import defaultdict


class FCFModel(object):
    """
    FriendBasedCF
    """
    def __init__(self, eta=0.5):
        self.eta = eta
        self.social_proximity = defaultdict(list)
        self.check_in_matrix = None
        self.pro_matrix = None

    def save_result(self, path, circle_pro_matrix):
        ctime = time.time()
        print("Saving FCF the result...")
        np.save(path + circle_pro_matrix, self.pro_matrix)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_result(self, path, circle_pro_matrix):
        ctime = time.time()
        print("Loading FCF result...", )
        self.pro_matrix = np.load(path + circle_pro_matrix + ".npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def compute_friend_sim(self, social_relations, social_friends, check_in_matrix):
        ctime = time.time()
        print("Precomputing FCF similarity between friends...", )
        self.check_in_matrix = check_in_matrix
        for uid, fids in social_relations.items():
            for fid in fids:
                u_social_neighbors = set(social_friends[uid])
                f_social_neighbors = set(social_friends[fid])
                if len(u_social_neighbors.union(f_social_neighbors)):
                    jaccard_friend = (1.0 * len(u_social_neighbors.intersection(f_social_neighbors)) /
                                      len(u_social_neighbors.union(f_social_neighbors)))
                else:
                    jaccard_friend = 0.0

                u_check_in_neighbors = set(check_in_matrix[int(uid), :].nonzero()[0])
                f_check_in_neighbors = set(check_in_matrix[int(fid), :].nonzero()[0])
                if (len(u_check_in_neighbors.union(f_check_in_neighbors))):
                    jaccard_check_in = (1.0 * len(u_check_in_neighbors.intersection(f_check_in_neighbors)) /
                                        len(u_check_in_neighbors.union(f_check_in_neighbors)))
                else:
                    jaccard_check_in = 0.0

                if jaccard_friend >= 0 and jaccard_check_in >= 0:
                    uid = int(uid)
                    self.social_proximity[uid].append([fid, jaccard_friend, jaccard_check_in])

        print("Done. Elapsed time:", time.time() - ctime, "s")

    def compute_pro_matrix(self, user_num, loc_num):
        ctime = time.time()
        print("Precomputing FCF scores...", )
        pro_matrix = np.zeros((user_num, loc_num))
        for i in range(user_num):
            for j in range(loc_num):
                if i in self.social_proximity:
                    pro_matrix[i, j] = np.sum(
                        [(self.eta * jf + (1 - self.eta) * jc) * self.check_in_matrix[int(k), j] for k, jf, jc in
                         self.social_proximity[i]])
                else:
                    pro_matrix[i, j] = 0.0
        self.pro_matrix = pro_matrix
        print("Done. Elapsed time:", time.time() - ctime, "s")
        return pro_matrix

    def predict(self, uid, lid):
        uid = int(uid)
        lid = int(lid)
        return self.pro_matrix[uid, lid]

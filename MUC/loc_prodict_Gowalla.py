# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from model.MFCModel import MFCModel
from model.UCFModel import UCFModel
from model.FCFModel import FCFModel
from model.LNSModel import LNSModel
from metric.metrics import accuracy


def read_friend_data():
    location_friends = open(user_location_friends_file, 'r').readlines()
    social_friends = open(user_social_friends_file, 'r').readlines()
    neighbor_friends = open(user_neighbor_friends_file, 'r').readlines()
    user_location_friends = defaultdict(list)
    user_social_friends = defaultdict(list)
    user_neighbor_friends = defaultdict(list)
    for eachline in location_friends:
        users = eachline.strip().split()
        user_id = users[0]
        for u in users[1:]:
            user_location_friends[user_id].append(u)

    for eachline in social_friends:
        users = eachline.strip().split()
        user_id = users[0]
        for u in users[1:]:
            user_social_friends[user_id].append(u)

    for eachline in neighbor_friends:
        users = eachline.strip().split()
        user_id = users[0]
        for u in users[1:]:
            user_neighbor_friends[user_id].append(u)
    return user_location_friends, user_social_friends, user_neighbor_friends


def read_training_data():
    train_data = open(train_file, 'r').readlines()
    training_matrix = np.zeros((user_num, loc_num))
    user_loc_matrix = np.zeros((user_num, loc_num))
    uid_lid_list = []
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        training_matrix[uid, lid] = freq
        user_loc_matrix[uid, lid] = 1.0
        uid_lid_list.append((uid, lid))
    return training_matrix, user_loc_matrix, uid_lid_list


def read_testing_truth():
    ground_truth = defaultdict(set)
    test_data = open(test_file, 'r').readlines()
    for eachline in test_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def read_user_feature():
    feature_data = np.loadtxt(user_features_file)
    feature_data_normalized1 = preprocessing.normalize(feature_data[:, 2:6], norm='l2')
    feature_data_normalized2 = preprocessing.normalize(feature_data[:, 7:11], norm='l2')
    feature_data1 = np.column_stack((feature_data[:, :2], feature_data_normalized1, feature_data[:, 6],
                                     feature_data_normalized2, feature_data[:, 11], feature_data_normalized2))
    return feature_data1


def normalize(scores):
    max_score = max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores


def main():
    training_matrix, user_loc_matrix, uid_lid_list = read_training_data()
    ground_truth = read_testing_truth()
    user_location_friends, user_social_friends, user_neighbor_friends = read_friend_data()
    user_features = read_user_feature()

    MFC.read_check_fre(training_matrix)

#     UCF.pre_compute_user_loc_scores(user_loc_matrix)
#     UCF.save_result("./tmp/")
    UCF.load_result("./tmp/")

#     FCF_LF.compute_friend_sim(user_location_friends, user_social_friends, user_loc_matrix)
#     LF = FCF_LF.compute_pro_matrix(user_num, loc_num)
#     FCF_LF.save_result("./tmp/", "LF_1_20")

#     FCF_NF.compute_friend_sim(user_neighbor_friends, user_social_friends, user_loc_matrix)
#     NF = FCF_NF.compute_pro_matrix(user_num, loc_num)
#     FCF_NF.save_result("./tmp/", "NF_1_20")

#     FCF_SF.compute_friend_sim(user_social_friends, user_social_friends, user_loc_matrix)
#     SF = FCF_SF.compute_pro_matrix(user_num, loc_num)
#     FCF_SF.save_result("./tmp/", "SF_1_20")

    FCF_LF.load_result("./tmp/", "LF_1_20")
    FCF_NF.load_result("./tmp/", "NF_1_20")
    FCF_SF.load_result("./tmp/", "SF_1_20")
#     LNS.sgd_lns(100, uid_lid_list, user_features, FCF_LF.pro_matrix, FCF_NF.pro_matrix, FCF_SF.pro_matrix)
#     LNS.save_result("./tmp/")
    LNS.load_result("./tmp/", FCF_LF.pro_matrix, FCF_NF.pro_matrix, FCF_SF.pro_matrix)

    all_uids = list(range(user_num))
    all_lids = list(range(loc_num))
    np.random.shuffle(all_uids)
    acc1 = 0.0
    acc2 = 0.0
    acc3 = 0.0
    for idx, uid in enumerate(all_uids):
        if uid in ground_truth:
            MFC_scores = normalize([MFC.predict(uid, lid) for lid in all_lids])
            UCF_scores = normalize([UCF.predict(uid, lid) for lid in all_lids])
            # FCF_LF_scores = normalize([FCF_LF.predict(uid, lid) for lid in all_lids])
            # FCF_NF_scores = normalize([FCF_NF.predict(uid, lid) for lid in all_lids])
            # FCF_SF_scores = normalize([FCF_SF.predict(uid, lid) for lid in all_lids])
            LNS_scores = normalize([LNS.predict(uid, lid) for lid in all_lids])

            MFC_scores = np.array(MFC_scores)
            UCF_scores = np.array(UCF_scores)
            # FCF_LF_scores = np.array(FCF_LF_scores)
            # FCF_NF_scores = np.array(FCF_NF_scores)
            # FCF_SF_scores = np.array(FCF_SF_scores)
            LNS_scores = np.array(LNS_scores)

            # overall_scores = MFC_scores
            overall_scores = alpha * (beta * MFC_scores + (1 - beta) * UCF_scores) + (1 - alpha) * LNS_scores
            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]

            acc1 += accuracy(actual, predicted[:1])
            acc2 += accuracy(actual, predicted[:2])
            acc3 += accuracy(actual, predicted[:3])

    print("alpha:", alpha)
    print("beta:", beta)
    print("unum:", len(ground_truth.keys()))
    print(len(ground_truth.keys()))
    Accuarcy1 = acc1 / len(ground_truth.keys())
    Accuarcy2 = acc2 / len(ground_truth.keys())
    Accuarcy3 = acc3 / len(ground_truth.keys())
    print("Accuarcy1:", Accuarcy1)
    print("Accuarcy2:", Accuarcy2)
    print("Accuarcy3:", Accuarcy3)


if __name__ == '__main__':
    data_dir = "../datasets/Gowalla/"

    size_file = data_dir + "Gowalla_data_size.txt"
    train_file = data_dir + "Gowalla_checkins_Train.txt"
    test_file = data_dir + "Gowalla_checkins_Test.txt"
    user_location_friends_file = data_dir + "Gowalla_location_friends.txt"
    user_neighbor_friends_file = data_dir + "Gowalla_neighbor_friends_20.txt"
    user_social_friends_file = data_dir + "Gowalla_social_friends_1.txt"
    user_features_file = data_dir + "Gowalla_user_features_1_20.txt"

    user_num, loc_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, loc_num = int(user_num), int(loc_num)
    top_k = 3
    alpha = 0.8
    beta = 0.8

    MFC = MFCModel()
    UCF = UCFModel()
    FCF_LF = FCFModel(eta=0.2)
    FCF_NF = FCFModel(eta=0.6)
    FCF_SF = FCFModel(eta=0.7)
    LNS = LNSModel(lamb=0.05)

    main()

# -*- coding: utf-8 -*-

class MFCModel(object):
	"""
	MFCModel
	"""
	def __init__(self):
		self.check_in_matrix = None

	def read_check_fre(self, training_matrix):
		self.check_in_matrix = training_matrix

	def predict(self, uid, lid):
		if self.check_in_matrix[uid, lid] == 0.0:
			return 0.0
		else:
			pro = self.check_in_matrix[uid, lid] / self.check_in_matrix[uid].sum()
		return pro


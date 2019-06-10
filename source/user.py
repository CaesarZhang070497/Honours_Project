import poll

import numpy as np
from collections import defaultdict


class User(object):
	"""docstring for User"""
	def __init__(self, userid):
		self.userid = userid
		self.lastActive = 0
		self.polls = {} #Key is pollId value is [interaction, index]

	def get_engagement_array(self, length):
		arr = np.zeros(length*5)
		for i in self.polls.keys():
			interaction = self.dec_to_binary_array(self.polls[i][0])
			idx = self.polls[i][1]
			arr[idx*5:idx*5+5] = interaction
		return arr

	def dec_to_binary_array(self, num):
		arr = np.zeros((5))
		# tmp = map(int, list(bin(num)[2:]))
		# arr[5-len(tmp):] = tmp
		tmp = map(int, list(bin(num)[2:]))
		arr[5-len(tmp):] = tmp
		return arr

	def add_interaction(self,pollid,interaction, pollidx):
		if pollid in self.polls.keys():
			self.polls[pollid][0] += interaction
		else:
			self.polls[pollid] = [interaction, pollidx]

	def num_interactions(self):
		return len(self.polls)

	def num_votes(self):
		votes = 0
		for i in self.polls.keys():
			bi = self.dec_to_binary_array(self.polls[i][0])
			votes += bi[-1]
		return votes
import json 
import user
import poll
import collections
import time
import numpy as np
class data_provider(object):
	"""docstring for data_provider"""
	def __init__(self, json_location):
		self.json_location = json_location
		self.polls = []
		self.users = collections.OrderedDict()
		self.hidden_interactions = []
	def parse(self):
		with open(self.json_location, 'r') as file:
		    content = file.read()
		data = json.loads(content)
		for i in list(set(data['users'].keys())):
			self.users[i] = user.User(i)
			self.users[i].lastActive = data['users'][i]["dateActivity"]
		for i in data['polls']:
			for j in data['polls'][i]:
				timestamp = np.random.uniform(low=(time.time()-24*3600), high=time.time()) # data['polls'][i][j]['creationDate'] # 
				title = data['polls'][i][j]['question']
				this = data['polls'][i][j]['this']
				that = data['polls'][i][j]['that']
				if j != "-LVcLeir9mEFVXOqDd30":
					p = poll.Poll(self.users[i].userid,j, timestamp, title, this, that)
					self.polls.append(p)
					self.users[i].add_interaction(j,8, self.polls.index(p))
				else:
					self.hidden_interactions.append([i,8])
					
		for i in data['votes'].keys(): #I = pollid
			if next((x for x in self.polls if x.pollid == i), None) == None:
				continue
			pollidx = self.polls.index(next((x for x in self.polls if x.pollid == i), None))
			for j in data['votes'][i].keys(): # J is userid 
				if j not in self.users.keys():
					self.users[j] = user.User(j)
				if j in self.users.keys() and u'skip' in data['votes'][i][j].keys():
					if i != "-LVcLeir9mEFVXOqDd30":
						self.users[j].add_interaction(i,16,pollidx)
					else:
						self.hidden_interactions.append([j,16])
				if j in self.users.keys() and u'this0that1removed2' in data['votes'][i][j].keys() and data['votes'][i][j][u'this0that1removed2'] in [0,1]:
					if i != "-LVcLeir9mEFVXOqDd30":
						self.users[j].add_interaction(i,1, pollidx)
					else:
						self.hidden_interactions.append([j,1])
					# self.users[j].polls[i] += 1 
		for i in data['commentsPath'].keys(): #i = userID
			for  j in data['commentsPath'][i].keys(): # j is pollId
				if next((x for x in self.polls if x.pollid == j), None) == None:
					continue
				pollidx = self.polls.index(next((x for x in self.polls if x.pollid == j), None))
				if j != "-LVcLeir9mEFVXOqDd30":
					self.users[i].add_interaction(j,2, pollidx)
				else:
					self.hidden_interactions.append([i,2])
				# self.users[i].polls[j] += 2
		for i in data['tracking'].keys(): #i = userID
			for  j in data['tracking'][i].keys(): # j is unknown
				for k in data['tracking'][i][j].keys():
					if next((x for x in self.polls if x.pollid == k), None) == None:
						continue
					pollidx = self.polls.index(next((x for x in self.polls if x.pollid == k), None))
					if i not in self.users.keys():
						self.users[i] = user.User(i)
					if k != "-LVcLeir9mEFVXOqDd30":
						self.users[i].add_interaction(k,4,pollidx)
					else:
						self.hidden_interactions.append([i,4])
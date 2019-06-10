

class Poll(object):
	"""docstring for Poll"""
	def __init__(self, userid, pollid, timestamp, title, this, that, boost=0):
		self.pollid = pollid
		self.userid = userid
		self.timestamp = timestamp
		self.title = title
		self.this = this
		self.that = that 
		self.boost = boost

	def get_text(self):
		return self.title + ' ' + self.this + '' + self.that
		
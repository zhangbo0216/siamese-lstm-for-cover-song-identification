import collections
import dan_tools
import numpy as np
import random

def padding(ar):
	max_length = 0
	res = []
	for i in ar:
		if len(i)>max_length:
			max_length = len(i)
	for i in ar:
		res.append(np.lib.pad(i, ((0, max_length - len(i)), (0,0)), 'constant', constant_values=(0)))        
	return np.array(res)

def sampling(ar, max_length):
	res = []
	for i in ar:
		res.append(i[:max_length])        
	return np.array(res)

class GetData(object):
	def __init__(self, path, size, max_length):
		with open(path) as f:
			content = f.readlines()
		content = [x.strip() for x in content] 
		song_dic = collections.defaultdict(list)
		key = ""
		for i in content:
			if i[0] == "%":
				key = i
			elif key:
				song_dic[key].append(i[:18])

		train_lable =  []
		train_name = []
		lable_count = 0
		for i in song_dic:
			for j in song_dic[i]:
				train_lable.append(lable_count)
				train_name.append(j)
			lable_count += 1
		train_samples = np.array(range(size))
		train_lable =  np.array(train_lable)
		train_name = np.array(train_name)
		sampled_lable = train_lable[train_samples]
		sampled_name = train_name[train_samples]

		PWR = 1.96
		low_level_feats = []
		lable = []
		er = 0
		song_id = [0]
		for i in range(len(sampled_name)):
			feats = dan_tools.msd_beatchroma("../SHS/"+sampled_name[i]+".h5")
			if feats != None:
				feats = dan_tools.chrompwr(feats, PWR)
				if feats != None:
					feats = feats.T
					low_level_feats.append(feats)
					lable += [sampled_lable[i]]
					song_id.append(song_id[-1]+len(feats))
				else:
					er += 1
			else:
				er += 1
		lable = np.array(lable)
		low_level_feats = sampling(low_level_feats, max_length)
		low_level_feats = padding(low_level_feats)
		x = []
		y = []
		z = []
		for i in range(len(low_level_feats)):
			for j in range(len(low_level_feats)):
				if i!= j:
					x.append(low_level_feats[i])
					y.append(low_level_feats[j])
					z.append(int(lable[i] == lable[j]))
					if random.random()>=0.01 and z[-1] == 0:
						x = x[:-1]
						y = y[:-1]
						z = z[:-1]
		self.train_set = [x, y, z]
		
	def get_train_set(self):
		return self.train_set

def get_500_queries(max_length):
	data = []
	with open('list_500queries.txt') as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	filenames = filter(lambda x: x[0]!='%', content)[1:]
	PWR = 1.96
	for i in filenames:
		feats = dan_tools.msd_beatchroma("../SHS/"+i+".h5")
		feats = dan_tools.chrompwr(feats, PWR)
		if feats!= None:
			feats = feats.T
			data.append(feats)
	data = sampling(data, max_length)
	data = padding(data)
	test_x1 = []
	test_x2 = []
	test_y = []
	for i in range(500):
		test_x1.append(data[3*i])
		test_x1.append(data[3*i])
		test_x2.append(data[3*i+1])
		test_x2.append(data[3*i+2])
		test_y.append(1)
		test_y.append(0)
	return [test_x1, test_x2, test_y]
        
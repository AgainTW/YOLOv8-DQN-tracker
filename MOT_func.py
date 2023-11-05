from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
import math

def distance(x1,x2):
	dist = math.pow(x1[0]-x2[0],2)+math.pow(x1[1]-x2[1],2)
	return dist

def MOT_table(cp, new_cp, KNN_flag):
	'''
	# cp[:,0,0] = label, 沒有使用就設為零, 同時使用 hash 的方式
	# cp[:,0,1] = 出現次數，用以清除長時間沒偵測到的 label
	'''
	cp_temp = np.zeros((20,3,4))


	empty_flag = 0 
	if( len(set(cp[:, 0, 0]))==1 and len(set(new_cp[:, 0, 0]))==1 ): empty_flag = 1 


	if(KNN_flag==0):
		for i in range(cp.shape[0]):
			cp_temp[i,:,:]	= cp[i,:,:]
	elif(empty_flag==1):
		cp_temp	= cp_temp
	else:
		# 變數設定
		x = []
		y = []
		for i in range(20):
			if(cp[i, 0, 0]!=0):
				x.append(cp[i, 1, 0:2])
				y.append(cp[i, 0, 0])
		x = np.array(x)
		y = np.array(y)
		new_x = new_cp[:,1,0:2]


		# 標籤預測
		clf = knn(1)
		clf.fit(x, y)
		new_y = clf.predict(new_x)

		# 處理"沒有"出現在預測的標籤
		### 找出沒有出現在預測(消失)的標籤
		disappear = list(set(range(1,21,1)).difference(set(new_y)))
		### 處理他們
		for i in disappear:
			if(cp[i-1, 0, 0]!=0):
				if(cp[i-1, 0, 1]>1):
					cp_temp[i-1, 0, 0] = i
					cp_temp[i-1, 0, 1] = cp[i-1, 0, 1] - 5
					cp_temp[i-1, 1:3, :] = cp[i-1, 1:3, :]
				else:
					cp_temp[i-1, :, :] = np.zeros((3,4))	
			else:
				cp_temp[i-1, :, :] = np.zeros((3,4))				

		# 處理有出現在預測的標籤
		### 利用 new_y 判斷 cp_temp 是否已經有相同 label，如果沒有則將 new_y 的標籤和 new_x 的值放入 cp_temp 
		### 如果 cp_temp 已經有相同 label 則以中心位置判斷，距離近的代表該 label
		### 距離遠者就在 cp_temp 中找空位，找不到就放棄該值
		for i in range(len(new_y)):
			label = new_y[i]
			if(cp_temp[label-1, 0, 0]!=0):
				dist_0 = distance(cp_temp[label-1, 1, 0:2], cp[label-1, 1, 0:2])
				dist_1 = distance(new_cp[i, 1, 0:2], cp[label-1, 1, 0:2])

				area_0 = cp_temp[label-1, 1, 2]*cp_temp[label-1, 1, 3]
				area_1 = new_cp[i, 1, 2]*new_cp[i, 1, 3]
				area_2 = cp[label-1, 1, 2]*cp[label-1, 1, 3]

				print(dist_0, (abs(area_0-area_2)/area_2))
				if( dist_0<=dist_1 or (abs(area_1-area_2)/area_2)>0.05):
					for j in disappear:
						if(cp_temp[j-1, 0, 0]==0):
							cp_temp[j-1, 0, 0] = j
							cp_temp[j-1, 0, 1] = 1
							cp_temp[j-1, 1:3, :] = new_cp[i, 1:3, :]
				else:
					for j in disappear:
						if(cp_temp[j-1, 0, 0]==0):
							cp_temp[j-1, 0, 0] = j
							cp_temp[j-1, 0, 1] = 1
							cp_temp[j-1, 1:3, :] = cp_temp[label-1, 1:3, :]
					cp_temp[label, 1:3, :] = new_cp[i, 1:3, :]
			else:
				cp_temp[label-1, 0, 0] = label
				if(0<cp[label-1, 0, 1]<50):
					cp_temp[label-1, 0, 1] = cp[label-1, 0, 1] + 1
				else:
					cp_temp[label-1, 0, 1] = 50
				cp_temp[label-1, 1:3, :] = new_cp[i, 1:3, :]				

	return cp_temp
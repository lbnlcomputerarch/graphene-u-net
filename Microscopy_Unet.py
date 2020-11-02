import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from unet import UNet
import os
import random
import time
import shutil
class Segmentation_Network:
	def __init__(self,D,W,reset_optim = False, model_name = 'UNET', LR = .0001, load = None,save_freq = 10):
		
		self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.D = D
		self.W = W
		self.load = load
		self.LR = LR
		self.save_freq = save_freq
		self.e = 0
		self.i = 0
		self.init_time = time.time()
		self.reset_optim = reset_optim
		self.Model_Init()
		#self.data = self.GetDataSet(data_path)
		self.model_name = model_name
		self.save_path=self.model_name

	def Model_Init(self):
 
		print('\ninitializing model')
		if self.load == None:
			print('>>>from blank slate')
			self.model = 0
			self.model = UNet(n_classes=2, padding=True, up_mode='upconv', depth=self.D,wf=self.W).to(self.device)
			self.optim = torch.optim.Adam(self.model.parameters(), self.LR)
		else:
			print('>>>from saved model')
			self.model = UNet(n_classes=2, padding=True, up_mode='upconv', depth=self.D,wf=self.W).to(self.device)
			self.optim = torch.optim.Adam(self.model.parameters(), self.LR)			
			self.load_model()
		self.criterion = nn.CrossEntropyLoss()


	def GetDataSet(self, data_path, handle = '_LABEL'):
		files = os.listdir(data_path)
		All_Data=[]	
		self.data_handle = handle
		for file in files:
			if self.data_handle in file:
				d = cv2.imread(data_path + file.split(self.data_handle)[0]+file.split(self.data_handle)[1],0)
				l = cv2.imread(data_path+file,0)
				#d = d-np.mean(d)
				#d = d/np.std(d)
				#d = self.PreProcess(d)
				l = (l>0).astype(int)
				All_Data.append((d,l,file))
		self.data = All_Data
        
	def DataShuffle(self, seed):
		random.seed(seed)
		random.shuffle(self.data)


	def init_progress_track(self):
		cols = 'time','epoch','F1_0','F1_1'
		self.metrics_df = pd.DataFrame(columns=cols)

	
	def update_progress_track(self):
		f1_0,f1_1=self.Validate()
		data_to_append = {'F1_0':f1_0, 'F1_1':f1_1, 'epoch':self.e, 'time':time.time()-self.init_time}
		
		self.metrics_df = self.metrics_df.append(data_to_append, ignore_index = True)
					

	def save_progress_track(self):
		self.metrics_df.to_csv(self.save_path+'.csv')

	def PreProcess(self,x):
		x = x-np.mean(x)
		x = x/np.std(x)
		return x
		
	def train(self,epochs=300):
		for e in range(epochs):
			self.e = e
			self.update_progress_track()
			self.save_progress_track()
			random.shuffle(self.training_data)
			for i in range(len(self.training_data)):
				self.i=i
				self.iterate()
			if e%self.save_freq==0:
				self.save_model()
					

	def Kfold(self, k=5, epochs=300):
		print('\n==========RUNNING '+str(k)+'-FOLD CROSS VALIDATION==========\n',end = '')
		fold_len = int(len(self.data)/5)
		sample=list(np.arange(0,len(self.data)))	
		for fold in range(k):
			print("\n ###############STARTING FOLD: "+str(fold)+" of "+ str(k)+"###############")
			self.Model_Init()
			self.init_progress_track()
			self.save_path=self.model_name+'_fold_'+str(fold)
			self.test_data = self.data[fold*fold_len:fold*fold_len+fold_len]
			self.training_data = self.data[:fold*fold_len]+self.data[(fold*fold_len+fold_len):]
			self.train(epochs)

	def iterate(self):
		x,gt,f = self.training_data[self.i]
		x = self.PreProcess(x)
		x = np.reshape(x,[1,1,x.shape[0],x.shape[1]])
		gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1]])
		x = torch.from_numpy(x).float()
		gt=torch.from_numpy(gt).long()
		x = x.to(self.device)
		gt = gt.to(self.device)
		self.optim.zero_grad()
		y = self.model(x)
		loss = self.criterion(y, gt)
		loss.backward()
		self.optim.step()
		self.loss = loss
		self.text ="\r"+'epoch:'+str(self.e)+'\t\tmodel:'+str(self.save_path)+ '\t\titeration:'+str(self.i)+'\t\tloss:'+str(round(loss.item(),5))
		sys.stdout.write(self.text)
		sys.stdout.flush()

	def Validate(self):
		print('\nEvaluating... ', end = '')
		l,w = (self.test_data[0][0]).shape
		z = len(self.test_data)
		#print(l,w,z)

		y_pred = np.zeros([l,w,z])
		y_true = np.zeros([l,w,z]) 
		images = []
		index = 0
		for x,gt,name in self.test_data:
			prediction = self.infer(x)
			y_pred[:,:,index]=prediction
			y_true[:,:,index]=gt
			index+=1
			images.append((x,prediction,gt))
		progr_folder = self.save_path+'_ep/'
		if not os.path.exists(progr_folder):
                        os.makedirs(progr_folder)

		check = self.stich(images)
		cv2.imwrite(progr_folder+str(self.e)+'.png',check)

		y_pred = np.ravel(y_pred)
		y_true = np.ravel(y_true)

		f1_0,f1_1 = metrics.f1_score(y_true=y_true,y_pred=y_pred,labels = [0,1],average=None)
		print("DICE class 0:",round(f1_0,3),"DICE class 1:",round(f1_1,3))
		return f1_0,f1_1

	def infer(self,x):
		x = self.PreProcess(x)
		x = np.reshape(x, [1,1,x.shape[0],x.shape[1]])
		x = torch.from_numpy(x).float()
		x = torch.cat((x, x), 0).to(self.device)
		y_out = self.model(x)
		y_out = y_out[0].squeeze().cpu()
		prediction = torch.argmax(y_out,dim=0).numpy()
		return prediction
	def to_int8(self,img):
		img = img-np.min(img)
		img = img/np.max(img)
		img = img*255
		img = img.astype(int)
		return img

	def stich(self, imgs):
		x,p,gt = imgs[0]
		x=self.to_int8(x)
		gt=self.to_int8(gt)
		p=self.to_int8(p)
		stitch=np.concatenate((x,p,gt),axis = 1)	
		for x,p,gt in imgs[1:]:
			x=self.to_int8(x)
			gt=self.to_int8(gt)
			p=self.to_int8(p)
			add =np.concatenate((x,p,gt),axis = 1)
			stitch = np.concatenate((stitch,add),axis = 0)
		return stitch
	
	def save_model(self):
		print(' --> saving model')
		torch.save({'epoch': self.e,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optim.state_dict(),'loss': self.loss}, self.save_path+'.pth')

	def load_model(self):
		print('loading model from', self.load)
		checkpoint = torch.load(self.load)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.e = 0
		self.l = 0
		if self.reset_optim==False:
			self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
		
			self.e = checkpoint['epoch']
			self.l = checkpoint['loss']
				
		print('reseting optimizer')



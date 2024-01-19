import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb
import os
import sys
import scipy.io
import mat73
import math
# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from util import log

def initialize_weights(module):
	if isinstance(module, nn.Conv2d):
		nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
	elif isinstance(module, nn.ConvTranspose2d):
		nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
	
	elif isinstance(module, nn.BatchNorm2d):
		module.weight.data.fill_(1)
		module.bias.data.zero_()
	elif isinstance(module, nn.Linear):
		module.bias.data.zero_()



def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

class Encoder(nn.Module):
	def __init__(self,in_dim):
		super(Encoder, self).__init__()

		
		self.encoder_N_units_per_FC_layer = 1024
		self.N_latent = 1024
		


		# self.conv1 = nn.Conv2d(1,8,kernel_size=3,stride=2, padding=0)
		
		self.fc1 = nn.Linear(in_dim, self.encoder_N_units_per_FC_layer)
		self.fc2 = nn.Linear(self.encoder_N_units_per_FC_layer, self.encoder_N_units_per_FC_layer)
		# self.dropout = nn.Dropout(0.5)
		self.linear_layer = nn.Linear(self.encoder_N_units_per_FC_layer, self.N_latent)
		
	   
		

	def forward(self, x):

	  
	  
		

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = torch.sigmoid(self.linear_layer(x))
		
	 
	   
	   
	  
		return x
def gen_gaussian_vector(sigma,x_coords,y_coords,device):


	x1, x2 = torch.meshgrid(torch.arange(0,300).to(device),
	                       torch.arange(0,300).to(device))
	x1 = x1.repeat(x_coords.shape[0],x_coords.shape[1],1,1)
	x2 = x2.repeat(y_coords.shape[0],y_coords.shape[1],1,1)
	

	# print(torch.from_numpy(x1).cuda().shape,x_coords.shape)
	y = (1/(torch.sqrt(torch.tensor([2*math.pi]).to(device))*sigma))*torch.exp(-0.5*((x1-x_coords.unsqueeze(2).unsqueeze(3))**2 + (x2-y_coords.unsqueeze(2).unsqueeze(3))**2)/(sigma**2))
	y_norm=(y-torch.amin(y,dim=(2,3)).unsqueeze(2).unsqueeze(3)) /((torch.amax(y,dim=(2,3))-torch.amin(y,dim=(2,3))).unsqueeze(2).unsqueeze(3))
	return y_norm



class analogy_dset(Dataset):
	def __init__(self, dset):
		self.ABC = dset['ABC']
		self.D_choices = dset['D_choices']
		self.D_correct = dset['D_correct']
		self.len = self.D_correct.shape[0]
	def __len__(self):
		return self.len
	def __getitem__(self, idx):
		ABC = self.ABC[idx,:]
		D_choices = self.D_choices[idx,:]
		D_correct = self.D_correct[idx]
		return ABC, D_choices, D_correct

class LSTM(nn.Module):
	def __init__(self, in_dim, num_layers=1):
		super(LSTM, self).__init__()
		log.info('Building LSTM and output layer...')
		self.in_dim = in_dim
		self.hidden_size = 512
		self.num_layers = num_layers
		# self.dropout = nn.Dropout(0.5)
		self.lstm = nn.LSTM(in_dim, self.hidden_size, self.num_layers, batch_first=True)
		self.score_out = nn.Linear(self.hidden_size,1)
	def forward(self, x_seq, device):

		# x_seq = x_seq * torch.unsqueeze(torch.unsqueeze(weights,0),0)
		# Initialize hidden state
		hidden = torch.zeros(self.num_layers, x_seq.shape[0], self.hidden_size).to(device)
		cell_state = torch.zeros(self.num_layers, x_seq.shape[0], self.hidden_size).to(device)
		# Apply LSTM
		lstm_out, (hidden, cell_state) = self.lstm(x_seq, (hidden, cell_state))
		lstm_final_out = lstm_out[:,-1,:]
		# Output layer
		score = self.score_out(lstm_final_out)
		return score

class MLP(nn.Module):
	def __init__(self, in_dim):
		super(MLP, self).__init__()
		log.info('Building MLP encoder...')
		self.in_dim = in_dim
		self.fc1 = nn.Linear(in_dim, 128)
		self.fc2 = nn.Linear(128, 128)
		self.relu = nn.ReLU()
	def forward(self, x):
		# Apply MLP
		fc1_out = self.relu(self.fc1(x))
		fc2_out = self.relu(self.fc2(fc1_out))
		return fc2_out

class scoring_model(nn.Module):
	def __init__(self, in_dim, mlp_encoder=False, multiLSTM=False):
		super(scoring_model, self).__init__()
		self.in_dim = in_dim

		self.weights=torch.empty(in_dim)
		self.weights=nn.init.normal_(self.weights)
		self.weights = torch.nn.Parameter(self.weights)

		self.mlp_encoder = mlp_encoder
		if multiLSTM:
			num_layers_LSTM = 2
		else:
			num_layers_LSTM = 1
		if self.mlp_encoder:
			self.mlp = MLP(in_dim)
			self.lstm = LSTM(128, num_layers=num_layers_LSTM)
		else:
			self.lstm = LSTM(in_dim, num_layers=num_layers_LSTM)
	def forward(self, ABC, D_choices,enc_model, device):
		# for i in range(self.in_dim//100):
		

			# ABC[:,:,i*100:(i+1)*100] = ABC[:,:,i*100:(i+1)*100].clone() * torch.sigmoid(self.weights[i])
			# D_choices[:,:,i*100:(i+1)*100] = D_choices[:,:,i*100:(i+1)*100].clone() * torch.sigmoid(self.weights[i])
		ABC = ABC * torch.unsqueeze(torch.unsqueeze(torch.sigmoid(self.weights),0),0)
		# ABC = ABC * torch.unsqueeze(torch.unsqueeze(torch.sigmoid(self.weights),0),0)
		# D_choices = D_choices * torch.unsqueeze(torch.unsqueeze(torch.sigmoid(self.weights),0),0)
		
		# ABC = ABC.clone() * torch.sigmoid(self.weights)[None,None,:]
		# D_choices = D_choices.clone() * torch.sigmoid(self.weights)[None,None,:]
			
		scores = []
		# Apply MLP encoder
		if self.mlp_encoder:
			ABC = self.mlp(ABC)
			D_choices = self.mlp(D_choices)
		# Loop through all choices and compute scores
		for d in range(D_choices.shape[1]):
			x_seq = torch.cat([ABC,enc_model(D_choices[:,d,:]).unsqueeze(1)* torch.unsqueeze(torch.unsqueeze(torch.sigmoid(self.weights),0),0)],dim=1)
			score = self.lstm(x_seq, device)
			scores.append(score)
		scores = torch.cat(scores,dim=1)
		return scores #, torch.sigmoid(self.weights)

def train(args, model,enc_model,sigma, device, train_loader,analogy_levels_loc,optimizer, epoch):
	# Set to training mode
	
	model.train()
	enc_model.train()
	# model2.train()
	# model3.train()
	# model4.train()
	# model5.train()
	# model6.train()
	# model7.train()
	
	# Iterate over batches
	all_loss = []
	all_acc = []
	all_log_det =[]
	for batch_idx, (ABC, D_choices, D_correct) in enumerate(train_loader):
		# Load data
		total_latent = torch.Tensor().to(device).float()
		ABC = ABC.to(device)
		D_choices = D_choices.to(device)
		D_correct = D_correct.to(device)
		
		# Convert to grid code
		# ABC = grid_codes[analogy_levels_loc[ABC],:]
		

		# ABC = grid_codes[analogy_levels_loc[ABC[:,:,0]],analogy_levels_loc[ABC[:,:,1]],:]
		# print("shape>>>",grid_codes[analogy_levels_loc[D_choices],:].shape)
		# D_choices = grid_codes[analogy_levels_loc[D_choices],:]
		# D_choices = grid_codes[analogy_levels_loc[D_choices[:,:,0]],analogy_levels_loc[D_choices[:,:,1]],:]
		ABC_gaussian = gen_gaussian_vector(sigma,analogy_levels_loc[ABC[:,:,0]],analogy_levels_loc[ABC[:,:,1]],device).float()
		# print ((ABC_onehots[0,0] == 1).nonzero(as_tuple=True)[0])
		# print ((ABC_onehots[0,1] == 1).nonzero(as_tuple=True)[0])
		# print ((ABC_onehots[0,2] == 1).nonzero(as_tuple=True)[0])
		
		# a=0.3

		# ABC_smoothed_onehots = (1 - a) * ABC_onehots + a/50000


		D_choices_gaussian = gen_gaussian_vector(sigma,analogy_levels_loc[D_choices[:,:,0]],analogy_levels_loc[D_choices[:,:,1]],device).float()
		A_latent = enc_model(torch.flatten(ABC_gaussian[:,0],start_dim=1,end_dim=2))
		B_latent = enc_model(torch.flatten(ABC_gaussian[:,1],start_dim=1,end_dim=2))
		C_latent = enc_model(torch.flatten(ABC_gaussian[:,2],start_dim=1,end_dim=2))
		# D_choices_smoothed_onehots = (1 - a) * D_choices_onehots + a/50000
		total_latent = torch.cat((total_latent,A_latent.unsqueeze(1),B_latent.unsqueeze(1),C_latent.unsqueeze(1)),dim=1)
		for d in range(D_choices.shape[1]):
			total_latent = torch.cat([total_latent,enc_model(torch.flatten(D_choices_gaussian,start_dim=2,end_dim=3)[:,d,:]).unsqueeze(1)],dim=1)
		total_latent_reshaped = total_latent.reshape(-1,total_latent.shape[2])
		total_latent_reshaped = torch.Tensor(np.unique(total_latent_reshaped.cpu().detach().numpy(),axis=0)).to(device)
		
		N= total_latent_reshaped.shape[0]
		M = total_latent_reshaped.shape[1]
		w_m = 1
		b=0.1
		eps=0.00001
		total_latent_reshaped = total_latent_reshaped +eps
		total_latent_reshaped = total_latent_reshaped/ torch.sqrt(torch.sum(total_latent_reshaped**2,1)).repeat(M).reshape(N,M)

		# print(samples_norm.shape)
		m = torch.diag(torch.tensor(np.cov(total_latent_reshaped.cpu().detach().numpy().T))).to(device).float()

		    # print(m.shape, np.cov(samples_norm.T).shape)
		L_scaled = torch.matmul(torch.transpose(total_latent_reshaped,0,1),total_latent_reshaped)
		M1 = torch.diag(torch.sqrt(torch.exp(w_m * m + b)))

		L_kernel = torch.matmul(M1, torch.matmul(L_scaled, M1))
		L_kernel = (L_kernel + torch.transpose(L_kernel,0,1))/2
		max_dets = torch.logdet(torch.matmul(torch.diag(torch.sigmoid(model.weights)), L_kernel- torch.eye(L_kernel.shape[0]).to(device)) + torch.eye(L_kernel.shape[0]).to(device))
			
		# print(ABC_onehots.shape,D_choices_onehots.shape)
		# Zero out gradients for optimizer 
		# temp_loss=[]
		# temp_acc=[]
		# temp_attention_weights=[]
		# print(ABC, ABC.shape)
		# print(D_choices, D_choices.shape)
		#model 1
		optimizer.zero_grad()
		# D_scores,attention_weights = model(ABC, D_choices, device)
		D_scores= model(torch.cat((A_latent.unsqueeze(1),B_latent.unsqueeze(1),C_latent.unsqueeze(1)),dim=1), torch.flatten(D_choices_gaussian,start_dim=2,end_dim=3),enc_model, device)
			# optimizer.zero_grad()
		# Scoring model
			# D_scores,attention_weights = model(ABC, D_choices, device)
		# Loss
		loss_fn = torch.nn.CrossEntropyLoss()
		# loss_kl = torch.nn.KLDivLoss()
		# print(loss_fn(D_scores, D_correct), loss_fn_mse(attention_weights,within_freq_dets_norm),within_freq_dets_norm)
		# loss = loss_fn(D_scores, D_correct) + args.lambda_* torch.sum(torch.mul(attention_weights,devs_freqwise)) #- args.lambda2_ * torch.sum(attention_weights)
		loss = loss_fn(D_scores, D_correct) - args.lambda_ * max_dets #+ args.lambda_* torch.sum(attention_weights) #- args.lambda2_ * torch.sum(attention_weights)
		

		# all_loss.append(temp_loss)
		# all_acc.append(temp_acc)

		all_loss.append(loss.item())
		all_log_det.append(max_dets.item())

		# Update model
		loss.backward()
		optimizer.step()
		# Accuracy
		D_pred = D_scores.argmax(1)
		acc = torch.eq(D_pred,D_correct).float().mean().item() * 100.0
		all_acc.append(acc)
		# Report progress
		if batch_idx % args.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
					 '[Loss = ' + '{:.4f}'.format(loss.item()) + '] ' + \
					 '[Log det = ' + '{:.4f}'.format(max_dets.item()) + '] ' + \
				 	 '[Accuracy = ' + '{:.2f}'.format(acc) + ']' )
			
			# print("Epoch =",epoch)
			# print("Batch  %s of %s "%(batch_idx,len(train_loader)))
			# print("Loss for each partition>>>",temp_loss)
			# print("Accuracy for each partition>>>",temp_acc)
			print("attention_weights>>>",torch.sigmoid(model.weights))
			# print("attention_weights>>>",attention_weights)
	# Average loss and accuracy
	all_loss = np.array(all_loss)
	all_acc = np.array(all_acc)
	return all_loss, all_acc


def val(args, model,enc_model,sigma, device, val_loader,analogy_levels_loc):
	# Set to training mode
	
	model.eval()
	enc_model.eval()
	# model2.train()
	# model3.train()
	# model4.train()
	# model5.train()
	# model6.train()
	# model7.train()
	
	# Iterate over batches
	all_loss = []
	all_acc = []
	for batch_idx, (ABC, D_choices, D_correct) in enumerate(val_loader):
		# Load data
	
		ABC = ABC.to(device)
		D_choices = D_choices.to(device)
		D_correct = D_correct.to(device)
		
		# Convert to grid code
		# ABC = grid_codes[analogy_levels_loc[ABC],:]
		

		# ABC = grid_codes[analogy_levels_loc[ABC[:,:,0]],analogy_levels_loc[ABC[:,:,1]],:]
		# print("shape>>>",grid_codes[analogy_levels_loc[D_choices],:].shape)
		# D_choices = grid_codes[analogy_levels_loc[D_choices],:]
		# D_choices = grid_codes[analogy_levels_loc[D_choices[:,:,0]],analogy_levels_loc[D_choices[:,:,1]],:]
		ABC_gaussian= gen_gaussian_vector(sigma,analogy_levels_loc[ABC[:,:,0]],analogy_levels_loc[ABC[:,:,1]],device).float()
		D_choices_gaussian= gen_gaussian_vector(sigma,analogy_levels_loc[D_choices[:,:,0]],analogy_levels_loc[D_choices[:,:,1]],device).float()
		A_latent = enc_model(torch.flatten(ABC_gaussian[:,0],start_dim=1,end_dim=2))
		B_latent = enc_model(torch.flatten(ABC_gaussian[:,1],start_dim=1,end_dim=2))
		C_latent = enc_model(torch.flatten(ABC_gaussian[:,2],start_dim=1,end_dim=2))

		# D_choices_smoothed_onehots = (1 - a) * D_choices_onehots + a/50000

		# print(ABC_onehots.shape,D_choices_onehots.shape)
		# Zero out gradients for optimizer 
		# temp_loss=[]
		# temp_acc=[]
		# temp_attention_weights=[]
		# print(ABC, ABC.shape)
		# print(D_choices, D_choices.shape)
		#model 1
	
		# D_scores,attention_weights = model(ABC, D_choices, device)
		D_scores = model(torch.cat((A_latent.unsqueeze(1),B_latent.unsqueeze(1),C_latent.unsqueeze(1)),dim=1),  torch.flatten(D_choices_gaussian,start_dim=2,end_dim=3), enc_model, device)
			# optimizer.zero_grad()
		# Scoring model
			# D_scores,attention_weights = model(ABC, D_choices, device)
		# Loss
		# loss_fn = torch.nn.CrossEntropyLoss()
		# loss_kl = torch.nn.KLDivLoss()
		# print(loss_fn(D_scores, D_correct), loss_fn_mse(attention_weights,within_freq_dets_norm),within_freq_dets_norm)
		# loss = loss_fn(D_scores, D_correct) + args.lambda_* torch.sum(torch.mul(attention_weights,devs_freqwise)) #- args.lambda2_ * torch.sum(attention_weights)
		# loss = loss_fn(D_scores, D_correct) #+ args.lambda_* torch.sum(attention_weights) #- args.lambda2_ * torch.sum(attention_weights)
		
		
		


		

		# all_loss.append(temp_loss)
		# all_acc.append(temp_acc)

	
		# Update model
	
		# Accuracy
		D_pred = D_scores.argmax(1)
		acc = torch.eq(D_pred,D_correct).float().mean().item() * 100.0
		all_acc.append(acc)
		# Report progress
		
			
			# print("Epoch =",epoch)
			# print("Batch  %s of %s "%(batch_idx,len(train_loader)))
			# print("Loss for each partition>>>",temp_loss)
			# print("Accuracy for each partition>>>",temp_acc)

			# print("attention_weights>>>",attention_weights)
	# Average loss and accuracy
	avg_acc = np.mean(all_acc)
	return avg_acc


def test(args,t,  model,enc_model,sigma, device, loader,analogy_levels_loc):
	# Set to evaluation mode
	model.eval()
	enc_model.eval()
	# Iterate over batches
	all_acc = []
	total_ABC_seqs= torch.Tensor().to(device).long()
	total_D_choices_seqs= torch.Tensor().to(device).long()
	total_D_correct_seqs= torch.Tensor().to(device).long()
	total_D_pred_seqs= torch.Tensor().to(device).long()
	
	
	for batch_idx, (ABC, D_choices, D_correct) in enumerate(loader):
		# Load data
		# ABC_onehots= torch.Tensor().to(device).float()
		# D_choices_onehots= torch.Tensor().to(device).float()
	
		# ABC_zeros = torch.zeros((ABC.shape[0],3,10000)).to(device).float()
		# D_choices_zeros = torch.zeros((D_choices.shape[0],7,10000)).to(device).float()
		
		ABC = ABC.to(device)
		D_choices = D_choices.to(device)
		D_correct = D_correct.to(device)

		# total_ABC_seqs = torch.cat((total_ABC_seqs,ABC))
		# total_D_choices_seqs = torch.cat((total_D_choices_seqs,D_choices))
		# total_D_correct_seqs = torch.cat((total_D_correct_seqs,D_correct))
		
		# Convert to grid code
		# ABC = grid_codes[analogy_levels_loc[ABC],:]
		# ABC = grid_codes[analogy_levels_loc[ABC[:,:,0]],analogy_levels_loc[ABC[:,:,1]],:]

		# D_choices = grid_codes[analogy_levels_loc[D_choices],:]

		# D_choices = grid_codes[analogy_levels_loc[D_choices[:,:,0]],analogy_levels_loc[D_choices[:,:,1]],:]
		ABC_gaussian= gen_gaussian_vector(sigma,analogy_levels_loc[ABC[:,:,0]],analogy_levels_loc[ABC[:,:,1]],device).float()
		D_choices_gaussian= gen_gaussian_vector(sigma,analogy_levels_loc[D_choices[:,:,0]],analogy_levels_loc[D_choices[:,:,1]],device).float()
		A_latent = enc_model(torch.flatten(ABC_gaussian[:,0],start_dim=1,end_dim=2))
		B_latent = enc_model(torch.flatten(ABC_gaussian[:,1],start_dim=1,end_dim=2))
		C_latent = enc_model(torch.flatten(ABC_gaussian[:,2],start_dim=1,end_dim=2))

		# for i in range(5):
		# 	if i==t:
		# 		ABC_onehots = torch.cat((ABC_onehots,ABC_onehots_onehalf),dim=2)
		# 		D_choices_onehots = torch.cat((D_choices_onehots,D_choices_onehots_onehalf),dim=2)

		# 	else:
		# 		ABC_onehots = torch.cat((ABC_onehots,ABC_zeros),dim=2)
		# 		D_choices_onehots = torch.cat((D_choices_onehots,D_choices_zeros),dim=2)


		# print
		# Scoring model
		# D_scores,_ = model(ABC, D_choices, device)
		# print(ABC_onehots.shape,D_choices_onehots.shape)
		
		# a=0.3
		# ABC_smoothed_onehots = (1 - a) * ABC_onehots + a/50000
		# D_choices_smoothed_onehots = (1 - a) * D_choices_onehots + a/50000

		D_scores = model(torch.cat((A_latent.unsqueeze(1),B_latent.unsqueeze(1),C_latent.unsqueeze(1)),dim=1),  torch.flatten(D_choices_gaussian,start_dim=2,end_dim=3), enc_model, device)

		# Accuracy
		D_pred = D_scores.argmax(1)
		# total_D_pred_seqs = torch.cat((total_D_pred_seqs,D_pred))
		
		acc = torch.eq(D_pred,D_correct).float().mean().item() * 100.0
		all_acc.append(acc)
	
	# if t>0:

	# 	print(total_ABC_seqs[:10],total_D_choices_seqs.gather(1,total_D_pred_seqs.view(-1,1))[:10],total_D_choices_seqs.gather(1,total_D_correct_seqs.view(-1,1))[:10])
	# Average accuracy
	avg_acc = np.mean(all_acc)
	return avg_acc

def save_nw(enc_model,model,optimizer,epoch, name):

	
	torch.save({
		'encoder_model_state_dict': enc_model.state_dict(),
		'model_state_dict': model.state_dict(),
		
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
	}, '/tigress/smondal/dpp_extrapolation/weights/'+name)
	
def load_checkpoint(enc_model,model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	enc_model.load_state_dict(model_ckp['encoder_model_state_dict'])
	model.load_state_dict(model_ckp['model_state_dict'])
	return enc_model,model	

def main():

	# Training settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_batch_size', type=int, default=35)
	parser.add_argument('--test_batch_size', type=int, default=35)
	parser.add_argument('--epochs', type=int, default=13)
	parser.add_argument('--log_interval', type=int, default=100)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--extrap_regime', type=str, default='translation', help="{'translation', 'scale'}")
	parser.add_argument('--partitions', type=str, default='diag', help="{'diag', 'xaxes','yaxes'}")
	
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	parser.add_argument('--lambda_', type=float, default=0.5)
	parser.add_argument('--sigma', type=float, default=1.0)
	parser.add_argument('--model_name', type=str, default='analogy_dppa_smoothed_onehots_encoder_lstm_module' )
	parser.add_argument('--model_checkpoint', type=str, default = '' )
	# parser.add_argument('--lambda2_', type=float, default=0.5)
	
	args = parser.parse_args()
	print(args)
	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Load analogy datasets
	log.info('Loading analogy datasets...')
	analogy_dset_dir = './analogy_dsets/' + args.extrap_regime + '/'
	# Training set
	train_set_files = np.load(analogy_dset_dir + 'train_{}_100x100_{}_2d_set.npz'.format(args.extrap_regime, args.partitions))  
	# test_region7_files = np.load(analogy_dset_dir + 'test_scale_set_6.npz')
	
	# train_set = {'ABC': train_set_files['ABC'],
	# 			 'D_choices': train_set_files['D_choices'],
	# 			 'D_correct': train_set_files['D_correct']}
	
	train_set = {'ABC': train_set_files['ABC'][:int(0.8*train_set_files['ABC'].shape[0])],
				 'D_choices': train_set_files['D_choices'][:int(0.8*train_set_files['ABC'].shape[0])],
				 'D_correct': train_set_files['D_correct'][:int(0.8*train_set_files['ABC'].shape[0])]}
	

	val_set = {'ABC': train_set_files['ABC'][int(0.8*train_set_files['ABC'].shape[0]):],
				 'D_choices': train_set_files['D_choices'][int(0.8*train_set_files['ABC'].shape[0]):],
				 'D_correct': train_set_files['D_correct'][int(0.8*train_set_files['ABC'].shape[0]):]}
	

	# Test sets
	test_sets = []
	N_test_sets = 2
	for d in range(N_test_sets):
		test_set_files = np.load(analogy_dset_dir + 'test_{}_100x100_{}_2d_set_'.format(args.extrap_regime, args.partitions) + str(d) + '.npz')
		test_set = {'ABC': test_set_files['ABC'],
					'D_choices': test_set_files['D_choices'],
					'D_correct': test_set_files['D_correct']}
		test_sets.append(test_set)
	# N levels
	# N_levels = int(np.load(analogy_dset_dir + 'N_levels.npz')['N_levels'])
	N_levels=1000
	# Get unique values for each dataset (for performing DPP attention/sorting)
	train_unique = np.unique(train_set['D_choices'])
	test_unique = []
	for t in range(len(test_sets)):
		test_unique.append(np.unique(test_sets[t]['D_choices']))
	# Convert to pytorch data loaders
	train_set = analogy_dset(train_set)

	train_loader = DataLoader(train_set, batch_size=args.train_batch_size, pin_memory = True,num_workers = 8,shuffle=True)
	
	val_set = analogy_dset(val_set)

	val_loader = DataLoader(val_set, batch_size=args.train_batch_size, pin_memory = True,num_workers = 8,shuffle=False)
	

	test_loaders = []
	for t in range(len(test_sets)):
		test_set = analogy_dset(test_sets[t])
		test_loaders.append(DataLoader(test_set, batch_size=args.test_batch_size,pin_memory = True, num_workers =8,shuffle=False))
	# ABC, D_choices, D_correct = next(iter(test_loaders[2]))
	print("Number of samples in training set>>",len(train_set))
	print("Number of samples in validation set>>",len(val_set))
	
	# Get grid code representations
	# log.info('Generating grid code...')
	# original_grid_codes = scipy.io.loadmat('/jukebox/griffiths/people/smondal/dpp_extrapolation/data/DOE_GC_FRmaps_100offsets.mat')['GC_FRmaps']
	# original_grid_codes = mat73.loadmat('/jukebox/griffiths/people/smondal/dpp_extrapolation/data/BB_gridcells_2D_1000x1000_b.mat')['GC_FRmaps']


	# original_grid_codes = np.transpose(original_grid_codes,(0,1,3,2)).reshape((original_grid_codes.shape[0],original_grid_codes.shape[1],original_grid_codes.shape[2]*original_grid_codes.shape[3]))

	# original_grid_codes = np.load('./grid_codes_8freqs.npz')['grid_codes']
	# N_loc = original_grid_codes.shape[0]
	# N_grid_cells = original_grid_codes.shape[2]
	
	# print("Number of grid cells>>>",N_grid_cells)

	# print("Number of locations and levels>>>", N_loc, N_levels)

	# original_grid_codes = torch.Tensor(original_grid_codes).to(device)
	# Convert analogy indices to grid code space
	N_loc=N_levels
	min_loc = 0
	analogy_levels_loc = torch.Tensor(np.arange(0,N_levels) * int(N_loc/N_levels) + min_loc).to(device).long()
	

	# Build model1
	log.info('Building model...')
	

	enc_model = Encoder(90000).to(device)

	model = scoring_model(1024, mlp_encoder=False, multiLSTM=False).to(device)
	# model= load_checkpoint(model,args.model_checkpoint)

	# Create optimizer1
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(list(enc_model.parameters())+list(model.parameters()), lr=args.lr)
	# optimizer = optim.SGD(model.parameters(),
	# 							   lr=args.lr,
	# 							   momentum=0.9,
	# 							   weight_decay=1e-4)
		

	
	



	# Model name
	
	model_name = args.model_name  #+ str(args.lambda_)

	
	# Apply attention to subset of grid cells for training set
	# grid_codes = deepcopy(original_grid_codes)

	
	# Train
	log.info('Training begins...')
	all_loss = []
	all_acc = []
	all_test_epochwise_acc = []
	test_loaders = [train_loader] + test_loaders
	# cells_include_test = [cells_include_train] + cells_include_test
	test_unique = [train_unique] + test_unique
	max_val_acc=0
	for epoch in range(1, args.epochs + 1):
		# Training loop
		loss, acc = train(args, model,enc_model,args.sigma, device, train_loader, analogy_levels_loc,optimizer, epoch)
		# all_loss.append(loss)
		# all_acc.append(acc)
		valacc = val(args, model,enc_model,args.sigma, device, val_loader,analogy_levels_loc)
		if valacc > max_val_acc:
			print("Average validation accuracy increased from %s to %s"%(max_val_acc,valacc))
			max_val_acc = valacc
			# print("Saving model$$$$$")
			# save_nw(enc_model,model,optimizer,epoch,'{}_{}_sigma_{}_run_{}_best.pth.tar'.format(args.model_name, args.extrap_regime,args.sigma,args.run))
			temp_test_acc=[]
			for t in range(len(test_loaders)):

				# Apply attention to subset of grid cells for training set
				# grid_codes = deepcopy(original_grid_codes)
				
				# Evaluate
				# print(t,cells_include_test[t])
			
				# print(t,len(test_loaders))
				acc = test(args,t, model,enc_model,args.sigma, device, test_loaders[t],analogy_levels_loc)
				log.info('Test region ' + str(t) + ' accuracy: ' + str(acc))
				temp_test_acc.append(acc)

		else:
			print("Average validation accuracy did not increase, skipping model saving!!!!!!!")

			
		print("Best validation accuracy till now>>",max_val_acc)
		print("Corresponding test accuracies>>",temp_test_acc)

			

			
	# Save training progress
	# train_prog_dir = './train_prog/'
	# check_path(train_prog_dir)
	# extrap_regime_dir = train_prog_dir + args.extrap_regime + '_extrap/'
	# check_path(extrap_regime_dir)
	# model_dir = extrap_regime_dir + model_name + '/'
	# check_path(model_dir)
	# train_prog_fname = model_dir + 'run' + args.run + '.npz'
	# np.savez(train_prog_fname,
	# 		 loss=np.array(all_loss),
	# 		 acc=np.array(all_acc))

	# Evaluate on training and test sets
	# test_loaders = [train_loader] + test_loaders
	# cells_include_test = [cells_include_train] + cells_include_test
	# test_unique = [train_unique] + test_unique
	# all_test_acc = []
	# for t in range(len(test_loaders)):
	# 	# Apply attention to subset of grid cells for training set
	# 	grid_codes = deepcopy(original_grid_codes)
	# 	cells_ignore = torch.logical_not(torch.any(torch.arange(N_grid_cells).unsqueeze(0) == torch.Tensor(cells_include_test[t]).unsqueeze(1),0)).to(device)
	# 	grid_codes[:,cells_ignore] = (grid_codes[:,cells_ignore] * 0) # - 1
	# 	# Evaluate
	# 	acc = test(args, model, device, test_loaders[t], grid_codes,cells_include_test[t], analogy_levels_loc)
	# 	log.info('Test region ' + str(t) + ' accuracy: ' + str(acc))
	# 	all_test_acc.append(acc)
	# Save results
	test_dir = './test/'
	check_path(test_dir)
	extrap_regime_dir = test_dir + args.extrap_regime + '_extrap/'
	check_path(extrap_regime_dir)
	model_dir = extrap_regime_dir + model_name + '/'
	check_path(model_dir)
	test_results_fname = model_dir + 'run' + args.run + '.npz'
	np.savez(test_results_fname,max_valacc = max_val_acc, acc=np.array(temp_test_acc))

if __name__ == '__main__':
	main()
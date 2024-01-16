import torch
import torch.nn as nn
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
from scipy.sparse import spdiags
from scipy.stats import norm
import h5py
import mat73


# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True

from encoder import *
from util import log

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

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
		ABCD = np.concatenate([ABC,np.expand_dims(D_choices[D_correct],0)],axis=0)
		not_D_choices = np.delete(D_choices,D_correct,0)
		return ABCD,not_D_choices,D_correct




class Context_norm_model(nn.Module):
	def __init__(self, latent_dim):
		super(Context_norm_model, self).__init__()
		self.latent_dim = latent_dim
		self.scale=torch.empty(latent_dim)
		self.scale=nn.init.ones_(self.scale)
		self.scale = torch.nn.Parameter(self.scale)

		self.shift=torch.empty(latent_dim)
		self.shift=nn.init.zeros_(self.shift)
		self.shift = torch.nn.Parameter(self.shift)



	
	def forward(self, A_latent, B_latent, C_latent, D_latent, ABCD_mean, ABCD_SD, device):
		A_context_norm = (((A_latent - ABCD_mean) / ABCD_SD) * self.scale.unsqueeze(0)) + self.shift.unsqueeze(0)
		B_context_norm = (((B_latent - ABCD_mean) / ABCD_SD) * self.scale.unsqueeze(0)) + self.shift.unsqueeze(0)
		C_context_norm = (((C_latent - ABCD_mean) / ABCD_SD) * self.scale.unsqueeze(0)) + self.shift.unsqueeze(0)
		D_context_norm = (((D_latent - ABCD_mean) / ABCD_SD) * self.scale.unsqueeze(0)) + self.shift.unsqueeze(0)

		return A_context_norm, B_context_norm, C_context_norm, D_context_norm

		

def train(args,grid_codes, enc_model, lstm_scoring_model,context_norm_model, device, train_loader,analogy_levels_loc,optimizer, epoch):
	# Set to training mode
	
	enc_model.train()
	lstm_scoring_model.train()
	context_norm_model.train()
	
	
	# Iterate over batches
	all_loss = []
	all_acc = []
	for batch_idx, (ABCD, not_D_choices, D_correct) in enumerate(train_loader):

		# Load data
		ABCD = ABCD.to(device)
		not_D_choices = not_D_choices.to(device)
		D_correct = D_correct.to(device)

		ABCD = grid_codes[analogy_levels_loc[ABCD[:,:,0]],analogy_levels_loc[ABCD[:,:,1]],:]
		
		A_latent = enc_model(ABCD[:,0])
		# log.info('B...')
		B_latent = enc_model(ABCD[:,1])
	# log.info('C...')
		C_latent = enc_model(ABCD[:,2])
		D_latent = enc_model(ABCD[:,3])

		not_D_choices = grid_codes[analogy_levels_loc[not_D_choices[:,:,0]],analogy_levels_loc[not_D_choices[:,:,1]],:]
		
		# log.info('D...')
		
		ABCD_mean = torch.mean(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),D_latent.unsqueeze(0)),dim=0),dim=0)
		ABCD_var = torch.var(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),D_latent.unsqueeze(0)),dim=0),dim=0)
		eps = 1e-8
		ABCD_SD = torch.sqrt(ABCD_var + eps)
		
		A_context_norm, B_context_norm, C_context_norm, D_context_norm = context_norm_model(A_latent, B_latent,C_latent, D_latent, ABCD_mean, ABCD_SD,device)
		

		N_foils = not_D_choices.shape[1]
		all_foil_latent = torch.Tensor().to(device).float()
		for foil in range(N_foils):
			all_foil_latent= torch.cat((all_foil_latent,enc_model(not_D_choices[:,foil]).unsqueeze(1)),dim=1)
	
		
		D_score = lstm_scoring_model(A_context_norm, B_context_norm, C_context_norm, D_context_norm,device)

		all_foil_score = torch.Tensor().to(device).float()
		for foil in range(N_foils):
			# Extract latent rep for this foil
			this_foil_latent = all_foil_latent[:,foil,:]
			# Get score

			ABCfoil_mean = torch.mean(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),this_foil_latent.unsqueeze(0)),dim=0),dim=0)
			ABCfoil_var = torch.var(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),this_foil_latent.unsqueeze(0)),dim=0),dim=0)
			eps = 1e-8
			ABCfoil_SD = torch.sqrt(ABCfoil_var + eps)
			A_context_norm, B_context_norm, C_context_norm, foil_context_norm = context_norm_model(A_latent, B_latent,C_latent, this_foil_latent, ABCfoil_mean, ABCfoil_SD,device)
		
			foil_score = lstm_scoring_model(A_context_norm, B_context_norm, C_context_norm, foil_context_norm,device)

			
			# Accumulate foil scores
			all_foil_score= torch.cat((all_foil_score,foil_score),dim=1)

		all_scores = torch.cat((D_score,all_foil_score),dim=1)
		# batch_shuffled_indices= torch.Tensor().to(device).long()
		# for i in range(A_latent.shape[0]):
			# batch_shuffled_indices= torch.cat((batch_shuffled_indices,torch.randperm(all_scores.shape[1]).to(device).unsqueeze(0)),dim=0)
		shuffled_choices = torch.randperm(all_scores.shape[1]).to(device)
		all_scores = all_scores[:,shuffled_choices]
		# all_scores = all_scores[:,batch_shuffled_indices]
		
		targets = torch.cat((torch.ones(D_score.shape).to(device), torch.zeros(all_foil_score.shape).to(device)), dim=1)
		targets = targets[:,shuffled_choices].argmax(1)
		# targets = targets[:,batch_shuffled_indices].argmax(1)
		
		# print(batch_idx,targets,batch_shuffled_indices)
		optimizer.zero_grad()
		
	
		# Loss
		loss_fn = torch.nn.CrossEntropyLoss()
		
	
		loss = loss_fn(all_scores, targets) 
		# loss = loss_fn(D_scores, D_correct) - args.lambda_* loss_kl(attention_weights,devs_cellwise) #- args.lambda2_ * torch.sum(attention_weights)
		
	


		

		# all_loss.append(temp_loss)
		# all_acc.append(temp_acc)

		all_loss.append(loss.item())
		# Update model
		loss.backward()
		optimizer.step()
		# Accuracy
		D_pred = all_scores.argmax(1)
		# equal=torch.Tensor().to(device).bool()
		# for c in range(1,all_scores.shape[1]):
		# 	# print(torch.eq(all_scores[:,0],all_scores[:,c]))
		# 	equal=torch.cat((equal,torch.eq(all_scores[:,0],all_scores[:,c]).unsqueeze(1)),dim=1)
		# # print(equal)
		# mask = torch.sum(torch.logical_not(equal).float(),dim=1).bool().float()

		acc = torch.eq(D_pred,targets).float().mean().item() * 100.0
		all_acc.append(acc)
		# Report progress
		if batch_idx % args.log_interval == 0:
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
					 '[Loss = ' + '{:.4f}'.format(loss.item()) + '] ' + \
					 '[Accuracy = ' + '{:.2f}'.format(acc) + ']' )
		
	# Average loss and accuracy
	all_loss = np.array(all_loss)
	all_acc = np.array(all_acc)
	return all_loss, all_acc

def val(args,grid_codes, enc_model,lstm_scoring_model,context_norm_model ,device, val_loader,analogy_levels_loc):
	# Set to evaluation mode
	enc_model.eval()
	lstm_scoring_model.eval()
	context_norm_model.eval()
	# Iterate over batches
	all_acc = []
	# total_ABC_seqs= torch.Tensor().to(device).long()
	# total_D_choices_seqs= torch.Tensor().to(device).long()
	# total_D_correct_seqs= torch.Tensor().to(device).long()
	
	total_test_scores= torch.Tensor().to(device).float()
				
	
			

	for batch_idx, (ABCD, not_D_choices, D_correct) in enumerate(val_loader):
		# Load data
		
		ABCD = ABCD.to(device)
		not_D_choices = not_D_choices.to(device)
		D_correct = D_correct.to(device)
		ABCD = grid_codes[analogy_levels_loc[ABCD[:,:,0]],analogy_levels_loc[ABCD[:,:,1]],:]

		# total_ABC_seqs = torch.cat((total_ABC_seqs,ABC))
		# total_D_choices_seqs = torch.cat((total_D_choices_seqs,D_choices))
		# total_D_correct_seqs = torch.cat((total_D_correct_seqs,D_correct))
		
		
		A_latent = enc_model(ABCD[:,0])
		# log.info('B...')
		B_latent = enc_model(ABCD[:,1])
	# log.info('C...')
		C_latent = enc_model(ABCD[:,2])
		# log.info('D...')
		D_latent = enc_model(ABCD[:,3])

		not_D_choices = grid_codes[analogy_levels_loc[not_D_choices[:,:,0]],analogy_levels_loc[not_D_choices[:,:,1]],:]
		# ABCD_mean, ABCD_var = tf.nn.moments(tf.stack([A_latent, B_latent, C_latent, D_latent], axis=0), 0)
		ABCD_mean = torch.mean(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),D_latent.unsqueeze(0)),dim=0),dim=0)
		ABCD_var = torch.var(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),D_latent.unsqueeze(0)),dim=0),dim=0)
		eps = 1e-8
		ABCD_SD = torch.sqrt(ABCD_var + eps)
		
		A_context_norm, B_context_norm, C_context_norm, D_context_norm = context_norm_model(A_latent, B_latent,C_latent, D_latent, ABCD_mean, ABCD_SD,device)
		
		N_foils = not_D_choices.shape[1]
		all_foil_latent = torch.Tensor().to(device).float()
		for foil in range(N_foils):
			all_foil_latent= torch.cat((all_foil_latent,enc_model(not_D_choices[:,foil]).unsqueeze(1)),dim=1)
	
		D_score = lstm_scoring_model(A_context_norm, B_context_norm, C_context_norm, D_context_norm,device)

		all_foil_score = torch.Tensor().to(device).float()
		for foil in range(N_foils):
			# Extract latent rep for this foil
			this_foil_latent = all_foil_latent[:,foil,:]
			# Get score

			ABCfoil_mean = torch.mean(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),this_foil_latent.unsqueeze(0)),dim=0),dim=0)
			ABCfoil_var = torch.var(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),this_foil_latent.unsqueeze(0)),dim=0),dim=0)
			eps = 1e-8
			ABCfoil_SD = torch.sqrt(ABCfoil_var + eps)
			A_context_norm, B_context_norm, C_context_norm, foil_context_norm = context_norm_model(A_latent, B_latent,C_latent, this_foil_latent, ABCfoil_mean, ABCfoil_SD,device)
		
			foil_score = lstm_scoring_model(A_context_norm, B_context_norm, C_context_norm, foil_context_norm,device)
			# Accumulate foil scores
			all_foil_score= torch.cat((all_foil_score,foil_score),dim=1)

		all_scores = torch.cat((D_score,all_foil_score),dim=1)
		# batch_shuffled_indices= torch.Tensor().to(device).long()
		# for i in range(A_latent.shape[0]):
			# batch_shuffled_indices= torch.cat((batch_shuffled_indices,torch.randperm(all_scores.shape[1]).to(device).unsqueeze(0)),dim=0)

		shuffled_choices = torch.randperm(all_scores.shape[1]).to(device)

		all_scores = all_scores[:,shuffled_choices]
		# all_scores = all_scores[batch_shuffled_indices]
		
		targets = torch.cat((torch.ones(D_score.shape).to(device), torch.zeros(all_foil_score.shape).to(device)), dim=1)
		targets = targets[:,shuffled_choices].argmax(1)
		# targets = targets[batch_shuffled_indices].argmax(1)
		
		# print(batch_idx,targets,batch_shuffled_indices)
		# equal=torch.Tensor().to(device).bool()
		# for c in range(1,all_scores.shape[1]):
		# 	equal=torch.cat((equal,torch.eq(all_scores[:,0],all_scores[:,c]).unsqueeze(1)),dim=1)
		
		# mask = torch.sum(torch.logical_not(equal).float(),dim=1).bool().float()

		# print(all_scores[:2])
		# if t>0:

		# 	total_test_scores=torch.cat((total_test_scores,all_scores[:1]),dim=0)
		
		# targets = torch.cat((torch.ones(D_score.shape).to(device), torch.zeros(all_foil_score.shape).to(device)), dim=1).argmax(1)
	
		# Accuracy
		D_pred = all_scores.argmax(1)
		# total_D_pred_seqs = torch.cat((total_D_pred_seqs,D_pred))
		
		acc = torch.eq(D_pred,targets).float().mean().item() * 100.0
		all_acc.append(acc)
	
	# if t>0:

	# 	print(total_ABC_seqs[:10],total_D_choices_seqs.gather(1,total_D_pred_seqs.view(-1,1))[:10],total_D_choices_seqs.gather(1,total_D_correct_seqs.view(-1,1))[:10])
	# Average accuracy
	avg_acc = np.mean(all_acc)
	# if t>0:

	# 	print("First 50 scores along dimensions>>>",total_test_scores[:50])
	return avg_acc 

def test(args,grid_codes, enc_model,lstm_scoring_model,context_norm_model ,device, loader,analogy_levels_loc):
	# Set to evaluation mode
	enc_model.eval()
	lstm_scoring_model.eval()
	context_norm_model.eval()
	# Iterate over batches
	all_acc = []
	# total_ABC_seqs= torch.Tensor().to(device).long()
	# total_D_choices_seqs= torch.Tensor().to(device).long()
	# total_D_correct_seqs= torch.Tensor().to(device).long()
	
	total_test_scores= torch.Tensor().to(device).float()
				
	
			

	for batch_idx, (ABCD, not_D_choices, D_correct) in enumerate(loader):
		# Load data
		
		ABCD = ABCD.to(device)
		not_D_choices = not_D_choices.to(device)
		D_correct = D_correct.to(device)
		ABCD = grid_codes[analogy_levels_loc[ABCD[:,:,0]],analogy_levels_loc[ABCD[:,:,1]],:]

		# total_ABC_seqs = torch.cat((total_ABC_seqs,ABC))
		# total_D_choices_seqs = torch.cat((total_D_choices_seqs,D_choices))
		# total_D_correct_seqs = torch.cat((total_D_correct_seqs,D_correct))
		
		
		A_latent = enc_model(ABCD[:,0])
		# log.info('B...')
		B_latent = enc_model(ABCD[:,1])
	# log.info('C...')
		C_latent = enc_model(ABCD[:,2])
		# log.info('D...')
		D_latent = enc_model(ABCD[:,3])

		not_D_choices = grid_codes[analogy_levels_loc[not_D_choices[:,:,0]],analogy_levels_loc[not_D_choices[:,:,1]],:]
		# ABCD_mean, ABCD_var = tf.nn.moments(tf.stack([A_latent, B_latent, C_latent, D_latent], axis=0), 0)
		ABCD_mean = torch.mean(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),D_latent.unsqueeze(0)),dim=0),dim=0)
		ABCD_var = torch.var(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),D_latent.unsqueeze(0)),dim=0),dim=0)
		eps = 1e-8
		ABCD_SD = torch.sqrt(ABCD_var + eps)
		
		A_context_norm, B_context_norm, C_context_norm, D_context_norm = context_norm_model(A_latent, B_latent,C_latent, D_latent, ABCD_mean, ABCD_SD,device)
		
		N_foils = not_D_choices.shape[1]
		all_foil_latent = torch.Tensor().to(device).float()
		for foil in range(N_foils):
			all_foil_latent= torch.cat((all_foil_latent,enc_model(not_D_choices[:,foil]).unsqueeze(1)),dim=1)
	
		D_score = lstm_scoring_model(A_context_norm, B_context_norm, C_context_norm, D_context_norm,device)

		all_foil_score = torch.Tensor().to(device).float()
		for foil in range(N_foils):
			# Extract latent rep for this foil
			this_foil_latent = all_foil_latent[:,foil,:]
			# Get score

			ABCfoil_mean = torch.mean(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),this_foil_latent.unsqueeze(0)),dim=0),dim=0)
			ABCfoil_var = torch.var(torch.cat((A_latent.unsqueeze(0),B_latent.unsqueeze(0),C_latent.unsqueeze(0),this_foil_latent.unsqueeze(0)),dim=0),dim=0)
			eps = 1e-8
			ABCfoil_SD = torch.sqrt(ABCfoil_var + eps)
			A_context_norm, B_context_norm, C_context_norm, foil_context_norm = context_norm_model(A_latent, B_latent,C_latent, this_foil_latent, ABCfoil_mean, ABCfoil_SD,device)
		
			foil_score = lstm_scoring_model(A_context_norm, B_context_norm, C_context_norm, foil_context_norm,device)
			# Accumulate foil scores
			all_foil_score= torch.cat((all_foil_score,foil_score),dim=1)

		all_scores = torch.cat((D_score,all_foil_score),dim=1)
		# batch_shuffled_indices= torch.Tensor().to(device).long()
		# for i in range(A_latent.shape[0]):
			# batch_shuffled_indices= torch.cat((batch_shuffled_indices,torch.randperm(all_scores.shape[1]).to(device).unsqueeze(0)),dim=0)

		shuffled_choices = torch.randperm(all_scores.shape[1]).to(device)

		all_scores = all_scores[:,shuffled_choices]
		# all_scores = all_scores[batch_shuffled_indices]
		
		targets = torch.cat((torch.ones(D_score.shape).to(device), torch.zeros(all_foil_score.shape).to(device)), dim=1)
		targets = targets[:,shuffled_choices].argmax(1)
		# targets = targets[batch_shuffled_indices].argmax(1)
		
		# print(batch_idx,targets,batch_shuffled_indices)
		# equal=torch.Tensor().to(device).bool()
		# for c in range(1,all_scores.shape[1]):
		# 	equal=torch.cat((equal,torch.eq(all_scores[:,0],all_scores[:,c]).unsqueeze(1)),dim=1)
		
		# mask = torch.sum(torch.logical_not(equal).float(),dim=1).bool().float()

		# print(all_scores[:2])
		# if t>0:

		# 	total_test_scores=torch.cat((total_test_scores,all_scores[:1]),dim=0)
		
		# targets = torch.cat((torch.ones(D_score.shape).to(device), torch.zeros(all_foil_score.shape).to(device)), dim=1).argmax(1)
	
		# Accuracy
		D_pred = all_scores.argmax(1)
		# total_D_pred_seqs = torch.cat((total_D_pred_seqs,D_pred))
		
		acc = torch.eq(D_pred,targets).float().mean().item() * 100.0
		all_acc.append(acc)
	
	# if t>0:

	# 	print(total_ABC_seqs[:10],total_D_choices_seqs.gather(1,total_D_pred_seqs.view(-1,1))[:10],total_D_choices_seqs.gather(1,total_D_correct_seqs.view(-1,1))[:10])
	# Average accuracy
	avg_acc = np.mean(all_acc)
	# if t>0:

	# 	print("First 50 scores along dimensions>>>",total_test_scores[:50])
	return avg_acc 

def save_nw(enc_model,lstm_scoring_model,context_norm_model,optimizer,epoch, name):

	
	torch.save({
		'encoder_model_state_dict': enc_model.state_dict(),
		'lstm_scoring_model_state_dict': lstm_scoring_model.state_dict(),
		'context_norm_model_state_dict': context_norm_model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
	}, '/tigress/smondal/dpp_extrapolation/weights/'+name)
	
def load_checkpoint(enc_model,lstm_scoring_model, context_norm_model,checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	enc_model.load_state_dict(model_ckp['encoder_model_state_dict'])
	lstm_scoring_model.load_state_dict(model_ckp['lstm_scoring_model_state_dict'])
	context_norm_model.load_state_dict(model_ckp['context_norm_model_state_dict'])
	
	return enc_model, lstm_scoring_model, context_norm_model	


	
def main():

	# Training settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_batch_size', type=int, default=32)
	parser.add_argument('--test_batch_size', type=int, default=32)
	parser.add_argument('--epochs', type=int, default=13)
	parser.add_argument('--partitions', type=str, default='diag', help="{'diag', 'xaxes','yaxes'}")
	
	parser.add_argument('--log_interval', type=int, default=100)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--extrap_regime', type=str, default='translation', help="{'translation', 'scale'}")
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	parser.add_argument('--lambda_', type=float, default=0.5)
	parser.add_argument('--model_name', type=str, default='analogy_tcn_gridcellcode_lstm_module' )
	# parser.add_argument('--model_checkpoint', type=str, default = '' )
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
	N_test_sets = 9
	for d in range(N_test_sets):
		test_set_files = np.load(analogy_dset_dir + 'test_{}_100x100_{}_2d_set_'.format(args.extrap_regime, args.partitions) + str(d) + '.npz')
		test_set = {'ABC': test_set_files['ABC'],
					'D_choices': test_set_files['D_choices'],
					'D_correct': test_set_files['D_correct']}
		test_sets.append(test_set)
	# N levels
	N_levels = 1000
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

	# Get grid code representations
	log.info('Generating grid code...')

	original_grid_codes = mat73.loadmat('BB_gridcells_2D_1000x1000_b.mat')['GC_FRmaps']
	N_freqs = original_grid_codes.shape[3]
	original_grid_codes = np.transpose(original_grid_codes,(0,1,3,2)).reshape((original_grid_codes.shape[0],original_grid_codes.shape[1],original_grid_codes.shape[2]*original_grid_codes.shape[3]))

	N_loc = original_grid_codes.shape[0]
	N_grid_cells = original_grid_codes.shape[2]
	
	print("Number of grid cells>>", N_grid_cells)

	print("Number of locations and levels>>>", N_loc, N_levels)

	original_grid_codes = torch.Tensor(original_grid_codes).to(device)
	# Convert analogy indices to grid code space
	min_loc = 0
	analogy_levels_loc = torch.Tensor(np.arange(0,N_levels) * int(N_loc/N_levels) + min_loc).to(device).long()

	log.info('Building model...')
	

	enc_model = Encoder(N_grid_cells).to(device)
	lstm_scoring_model = LSTM(260).to(device)
	context_norm_model = Context_norm_model(256).to(device)
	# model= load_checkpoint(model,args.model_checkpoint)

	
	# Create optimizer1
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(list(enc_model.parameters())+list(lstm_scoring_model.parameters()) +list(context_norm_model.parameters()), lr=args.lr)
	# optimizer = optim.SGD(model.parameters(),
	# 							   lr=args.lr,
	# 							   momentum=0.9,
	# 							   weight_decay=1e-4)
		

	model_name = args.model_name 

	
	# Apply attention to subset of grid cells for training set
	grid_codes = deepcopy(original_grid_codes)

	
	# Train
	log.info('Training begins...')
	all_loss = []
	all_acc = []
	all_test_epochwise_acc = []
	test_loaders = [train_loader] + test_loaders
	# cells_include_test = [cells_include_train] + cells_include_test
	
	max_val_acc=0
	
	flag=0
	for epoch in range(1, args.epochs + 1):
		
		# grid_codes = deepcopy(original_grid_codes)

		# Training loop
		
		
		loss, acc= train(args,grid_codes, enc_model,lstm_scoring_model,context_norm_model, device, train_loader,analogy_levels_loc,optimizer, epoch)
		
		all_loss.append(loss)
		all_acc.append(acc)

		valacc= val(args,grid_codes, enc_model,lstm_scoring_model,context_norm_model, device, val_loader,analogy_levels_loc)
		if valacc> max_val_acc:
			print("Average validation accuracy increased from %s to %s"%(max_val_acc,valacc))
			max_val_acc = valacc
			# print("Saving model$$$$$")
			# save_nw(enc_model,lstm_scoring_model,context_norm_model,optimizer,epoch,'{}_run_{}_best.pth.tar'.format(args.model_name,args.run))

			temp_test_acc=[]
			for t in range(len(test_loaders)):

			
			

				acc= test(args,grid_codes, enc_model,lstm_scoring_model,context_norm_model, device, test_loaders[t],analogy_levels_loc)
				# log.info('Test region ' + str(t) + ' accuracy: ' + str(acc))
				log.info('Test region ' + str(t) + ' accuracy: ' + str(acc))
				temp_test_acc.append(acc)

			
			
		# all_test_epochwise_acc.append(temp_test_acc)
		
		else:
			print("Average validation accuracy did not increase, skipping model saving!!!!!!!")
		print("Best validation accuracy is %s, and corresponding test accuracies %s"%(max_val_acc,temp_test_acc))

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
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
import mat73

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True


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
		return ABC, D_choices, D_correct

class LSTM(nn.Module):
	def __init__(self, in_dim, num_layers=1):
		super(LSTM, self).__init__()
		log.info('Building LSTM and output layer...')
		self.in_dim = in_dim
		self.hidden_size = 512
		self.num_layers = num_layers
		self.lstm = nn.LSTM(in_dim, self.hidden_size, self.num_layers, batch_first=True)
		self.score_out = nn.Linear(self.hidden_size,1)
	def forward(self, x_seq, device):
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
	def __init__(self,in_dim, mlp_encoder=False, multiLSTM=False):
		super(scoring_model, self).__init__()
		# self.weights=torch.empty(in_dim)
		# self.weights=nn.init.normal_(self.weights)
		# self.weights = torch.nn.Parameter(self.weights)


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
	def forward(self, ABC, D_choices, device):
	
		
		# ABC = ABC * torch.unsqueeze(torch.unsqueeze(torch.sigmoid(self.weights),0),0)
		# D_choices = D_choices * torch.unsqueeze(torch.unsqueeze(torch.sigmoid(self.weights),0),0)
		

		scores = []
		# Apply MLP encoder
		if self.mlp_encoder:
			ABC = self.mlp(ABC)
			D_choices = self.mlp(D_choices)
		# Loop through all choices and compute scores
		for d in range(D_choices.shape[1]):
			x_seq = torch.cat([ABC,D_choices[:,d,:].unsqueeze(1)],dim=1)
			score = self.lstm(x_seq, device)
			scores.append(score)
		scores = torch.cat(scores,dim=1)
		return scores  #, torch.sigmoid(self.weights) 

def train(args, model, device, train_loader, grid_codes, analogy_levels_loc,optimizer, epoch):
	# Set to training mode
	
	model.train()
	# model2.train()
	# model3.train()
	# model4.train()
	# model5.train()
	# model6.train()
	# model7.train()
	
	# Iterate over batches
	all_loss = []
	all_acc = []
	for batch_idx, (ABC, D_choices, D_correct) in enumerate(train_loader):

		# Load data
		ABC = ABC.to(device)
		D_choices = D_choices.to(device)
		D_correct = D_correct.to(device)
		
		# Convert to grid code
		# ABC = grid_codes[analogy_levels_loc[ABC],:]

		ABC = grid_codes[analogy_levels_loc[ABC[:,:,0]],analogy_levels_loc[ABC[:,:,1]],:]
		# print("shape>>>",grid_codes[analogy_levels_loc[D_choices],:].shape)
		# D_choices = grid_codes[analogy_levels_loc[D_choices],:]
		D_choices = grid_codes[analogy_levels_loc[D_choices[:,:,0]],analogy_levels_loc[D_choices[:,:,1]],:]
		
		# Zero out gradients for optimizer 
		# temp_loss=[]
		# temp_acc=[]
		# temp_attention_weights=[]
		# print(ABC, ABC.shape)
		# print(D_choices, D_choices.shape)
		#model 1
		optimizer.zero_grad()
		D_scores= model(ABC, D_choices, device)
		# D_scores= model(ABC, D_choices, device)

			# optimizer.zero_grad()
		# Scoring model
			# D_scores,attention_weights = model(ABC, D_choices, device)
		# Loss
		loss_fn = torch.nn.CrossEntropyLoss()
		# kl_loss = torch.nn.KLDivLoss()
		# loss_kl = torch.nn.KLDivLoss()
		# print(loss_fn(D_scores, D_correct), loss_fn_mse(attention_weights,within_freq_dets_norm),within_freq_dets_norm)
		

	
		# loss = loss_fn(D_scores, D_correct) 
		loss = loss_fn(D_scores, D_correct) #+ args.lambda_ * torch.sum(attention_weights)
		# loss = loss_fn(D_scores, D_correct) + args.lambda_ * kl_loss(attention_weights,1e-8*torch.ones(attention_weights.shape[0]).to(device))
		
	


		

		# all_loss.append(temp_loss)
		# all_acc.append(temp_acc)

		all_loss.append(loss.item())
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
				 	 '[Accuracy = ' + '{:.2f}'.format(acc) + ']' )
			
			# print("Epoch =",epoch)
			# print("Batch  %s of %s "%(batch_idx,len(train_loader)))
			# print("Loss for each partition>>>",temp_loss)
			# print("Accuracy for each partition>>>",temp_acc)
			# print("attention_weights>>>",attention_weights)
			
	all_loss = np.array(all_loss)
	all_acc = np.array(all_acc)
	return all_loss, all_acc 

def val(args,model, device, val_loader, grid_codes, analogy_levels_loc):
	# Set to evaluation mode
	model.eval()
	# Iterate over batches
	all_acc = []
	total_ABC_seqs= torch.Tensor().to(device).long()
	total_D_choices_seqs= torch.Tensor().to(device).long()
	total_D_correct_seqs= torch.Tensor().to(device).long()
	total_D_pred_seqs= torch.Tensor().to(device).long()
				
	# test_original_grid_codes = grid_codes.cpu().detach().numpy()[250*t:250*(t+1),250*t:250*(t+1)]
	# flattened_gc_maps_1D = test_original_grid_codes.reshape((-1,test_original_grid_codes.shape[2]))
	
	# N= flattened_gc_maps_1D.shape[0]
	# M = flattened_gc_maps_1D.shape[1]
	# w_m = 1
	# b=0.1
	# eps=0.00001
	# flattened_gc_maps_1D = flattened_gc_maps_1D +eps
	# flattened_gc_maps_1D_norm = flattened_gc_maps_1D / np.repeat(np.sqrt(np.sum(flattened_gc_maps_1D**2,axis =1)),M).reshape(N,M)

	# # print(samples_norm.shape)
	# m = np.diag(np.cov(flattened_gc_maps_1D_norm.T))
	#     # print(m.shape, np.cov(samples_norm.T).shape)
	# L_scaled = np.matmul(np.transpose(flattened_gc_maps_1D_norm), flattened_gc_maps_1D_norm)
	# M1 = spdiags(np.sqrt(np.exp(w_m * m + b)),0,M,M).toarray()

	# L_kernel = np.matmul(M1, np.matmul(L_scaled, M1))
	# L_kernel = (L_kernel + np.transpose(L_kernel))/2
	# L_kernel = torch.Tensor(L_kernel).to(device)
			
			

	for batch_idx, (ABC, D_choices, D_correct) in enumerate(val_loader):
		# Load data
		
		ABC = ABC.to(device)
		D_choices = D_choices.to(device)
		D_correct = D_correct.to(device)

		total_ABC_seqs = torch.cat((total_ABC_seqs,ABC))
		total_D_choices_seqs = torch.cat((total_D_choices_seqs,D_choices))
		total_D_correct_seqs = torch.cat((total_D_correct_seqs,D_correct))
		
		# Convert to grid code
		# ABC = grid_codes[analogy_levels_loc[ABC],:]
		ABC = grid_codes[analogy_levels_loc[ABC[:,:,0]],analogy_levels_loc[ABC[:,:,1]],:]

		# D_choices = grid_codes[analogy_levels_loc[D_choices],:]

		D_choices = grid_codes[analogy_levels_loc[D_choices[:,:,0]],analogy_levels_loc[D_choices[:,:,1]],:]
		
		# Scoring model
		D_scores = model(ABC, D_choices, device)
		
		# D_scores = model(ABC, D_choices, device)
		
		# within_freq_dets = []
		# for i in range(N_freqs):

		# 	within_freq_dets.append(torch.abs(torch.det(torch.matmul(torch.diag(attention_weights[i*100:(i+1)*100]), L_kernel[i*100:(i+1)*100,:][:,i*100:(i+1)*100]- torch.eye(L_kernel[i*100:(i+1)*100,:][:,i*100:(i+1)*100].shape[0]).cuda()) + torch.eye(L_kernel[i*100:(i+1)*100,:][:,i*100:(i+1)*100].shape[0]).cuda())).item())

		# Accuracy
		D_pred = D_scores.argmax(1)
		total_D_pred_seqs = torch.cat((total_D_pred_seqs,D_pred))
		
		acc = torch.eq(D_pred,D_correct).float().mean().item() * 100.0
		all_acc.append(acc)
	
	# if t>0:

	# 	print(total_ABC_seqs[:10],total_D_choices_seqs.gather(1,total_D_pred_seqs.view(-1,1))[:10],total_D_choices_seqs.gather(1,total_D_correct_seqs.view(-1,1))[:10])
	# Average accuracy
	avg_acc = np.mean(all_acc)
	return avg_acc #,within_
def test(args,t,  model, device, loader, grid_codes, analogy_levels_loc):
	# Set to evaluation mode
	model.eval()
	# Iterate over batches
	all_acc = []
	total_ABC_seqs= torch.Tensor().to(device).long()
	total_D_choices_seqs= torch.Tensor().to(device).long()
	total_D_correct_seqs= torch.Tensor().to(device).long()
	total_D_pred_seqs= torch.Tensor().to(device).long()
				
	# test_original_grid_codes = grid_codes.cpu().detach().numpy()[250*t:250*(t+1),250*t:250*(t+1)]
	# flattened_gc_maps_1D = test_original_grid_codes.reshape((-1,test_original_grid_codes.shape[2]))
	
	# N= flattened_gc_maps_1D.shape[0]
	# M = flattened_gc_maps_1D.shape[1]
	# w_m = 1
	# b=0.1
	# eps=0.00001
	# flattened_gc_maps_1D = flattened_gc_maps_1D +eps
	# flattened_gc_maps_1D_norm = flattened_gc_maps_1D / np.repeat(np.sqrt(np.sum(flattened_gc_maps_1D**2,axis =1)),M).reshape(N,M)

	# # print(samples_norm.shape)
	# m = np.diag(np.cov(flattened_gc_maps_1D_norm.T))
	#     # print(m.shape, np.cov(samples_norm.T).shape)
	# L_scaled = np.matmul(np.transpose(flattened_gc_maps_1D_norm), flattened_gc_maps_1D_norm)
	# M1 = spdiags(np.sqrt(np.exp(w_m * m + b)),0,M,M).toarray()

	# L_kernel = np.matmul(M1, np.matmul(L_scaled, M1))
	# L_kernel = (L_kernel + np.transpose(L_kernel))/2
	# L_kernel = torch.Tensor(L_kernel).to(device)
			
			

	for batch_idx, (ABC, D_choices, D_correct) in enumerate(loader):
		# Load data
		
		ABC = ABC.to(device)
		D_choices = D_choices.to(device)
		D_correct = D_correct.to(device)

		total_ABC_seqs = torch.cat((total_ABC_seqs,ABC))
		total_D_choices_seqs = torch.cat((total_D_choices_seqs,D_choices))
		total_D_correct_seqs = torch.cat((total_D_correct_seqs,D_correct))
		
		# Convert to grid code
		# ABC = grid_codes[analogy_levels_loc[ABC],:]
		ABC = grid_codes[analogy_levels_loc[ABC[:,:,0]],analogy_levels_loc[ABC[:,:,1]],:]

		# D_choices = grid_codes[analogy_levels_loc[D_choices],:]

		D_choices = grid_codes[analogy_levels_loc[D_choices[:,:,0]],analogy_levels_loc[D_choices[:,:,1]],:]
		
		# Scoring model
		D_scores = model(ABC, D_choices, device)
		
		# D_scores = model(ABC, D_choices, device)
		
		# within_freq_dets = []
		# for i in range(N_freqs):

		# 	within_freq_dets.append(torch.abs(torch.det(torch.matmul(torch.diag(attention_weights[i*100:(i+1)*100]), L_kernel[i*100:(i+1)*100,:][:,i*100:(i+1)*100]- torch.eye(L_kernel[i*100:(i+1)*100,:][:,i*100:(i+1)*100].shape[0]).cuda()) + torch.eye(L_kernel[i*100:(i+1)*100,:][:,i*100:(i+1)*100].shape[0]).cuda())).item())

		# Accuracy
		D_pred = D_scores.argmax(1)
		total_D_pred_seqs = torch.cat((total_D_pred_seqs,D_pred))
		
		acc = torch.eq(D_pred,D_correct).float().mean().item() * 100.0
		all_acc.append(acc)
	
	# if t>0:

	# 	print(total_ABC_seqs[:10],total_D_choices_seqs.gather(1,total_D_pred_seqs.view(-1,1))[:10],total_D_choices_seqs.gather(1,total_D_correct_seqs.view(-1,1))[:10])
	# Average accuracy
	avg_acc = np.mean(all_acc)
	return avg_acc #,within_freq_dets

def save_nw(model,optimizer,epoch, name):

	
	torch.save({
		'model_state_dict': model.state_dict(),
		
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
	}, '/tigress/smondal/dpp_extrapolation/weights/'+name)
	

def load_checkpoint(model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	model.load_state_dict(model_ckp['model_state_dict'])
	return model	

def weights_init(m):
    if isinstance(m, nn.Parameter):
        torch.nn.init.normal_(m.weight.data)
	
def main():

	# Training settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_batch_size', type=int, default=35)
	parser.add_argument('--test_batch_size', type=int, default=35)
	parser.add_argument('--epochs', type=int, default=13)

	
	parser.add_argument('--log_interval', type=int, default=500)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--extrap_regime', type=str, default='translation', help="{'translation', 'scale'}")
	parser.add_argument('--partitions', type=str, default='diag', help="{'diag', 'xaxes','yaxes'}")
	
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--run', type=str, default='1')
	
	parser.add_argument('--model_name', type=str, default='analogy_gridcellcode_lstm_module' )
	# parser.add_argument('--model_checkpoint', type=str, default = '' )
	# parser.add_argument('--lambda_', type=float, default=0.5)
	
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
	# N_levels = int(np.load(analogy_dset_dir + 'N_levels.npz')['N_levels'])
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
	# original_grid_codes = scipy.io.loadmat('/jukebox/griffiths/people/smondal/dpp_extrapolation/data/DOE_GC_FRmaps_random.mat')['GC_FRmaps']
	original_grid_codes = mat73.loadmat('BB_gridcells_2D_1000x1000_b.mat')['GC_FRmaps']
	# original_grid_codes = np.random.rand(1000,1000,100,9)

	N_freqs = original_grid_codes.shape[3]
	original_grid_codes = np.transpose(original_grid_codes,(0,1,3,2)).reshape((original_grid_codes.shape[0],original_grid_codes.shape[1],original_grid_codes.shape[2]*original_grid_codes.shape[3]))

	

	# original_grid_codes = np.load('./grid_codes_8freqs.npz')['grid_codes']
	N_loc = original_grid_codes.shape[0]
	N_grid_cells = original_grid_codes.shape[2]
	
	print("Number of grid cells>>", N_grid_cells)

	print("Number of locations and levels>>>", N_loc, N_levels)

	original_grid_codes = torch.Tensor(original_grid_codes).to(device)
	# Convert analogy indices to grid code space
	min_loc = 0
	analogy_levels_loc = torch.Tensor(np.arange(0,N_levels) * int(N_loc/N_levels) + min_loc).to(device).long()

	
	
	# devs_cellwise_total_cum = torch.cat((devs_cellwise_cum1,devs_cellwise_cum2,devs_cellwise_cum3,devs_cellwise_cum4,devs_cellwise_cum5,devs_cellwise_cum6,devs_cellwise_cum7),dim=0)
	# devs_freqwise = torch.tensor([1.        , 1.        , 1.        , 1.        , 1.        ,
 #       1.        , 0.        , 0.99355873, 0.99999989 ]).to(device)
	# train_withinfreq_dets = torch.tensor([6.2047e-27, 3.6554e-32, 4.6386e-39, 1.8217e-44, 8.5567e-26, 2.3358e-12,
 #        1.9779e+10, 1.2334e+10, 1.5305e+10]).to(device)
	# within_freq_dets_norm = (within_freq_dets - torch.min(within_freq_dets))/(torch.max(within_freq_dets)-torch.min(within_freq_dets))
	# Cells to include in training and test
	# cells_include_train = [0, 1, 2, 4, 7, 8, 9, 11, 12, 14, 15]
	# cells_include_test = [[0, 1, 2, 4, 7, 8, 9, 11, 12, 14, 15],
	# 					  [0, 1, 2, 4, 7, 8, 9, 11, 12, 14, 15],
	# 					  [0, 1, 2, 4, 7, 8, 9, 11, 12, 14, 15],
	# 					  [0, 1, 2, 4, 7, 8, 9, 11, 12, 14, 15],
	# 					  [0, 1, 2, 4, 7, 8, 9, 11, 12, 14, 15],
	# 					  [0, 1, 2, 4, 7, 8, 9, 11, 12, 14, 15]]


	# Build model1
	log.info('Building model...')
	

	model = scoring_model(N_grid_cells, mlp_encoder=False, multiLSTM=False).to(device)
	
	# model= load_checkpoint(model,args.model_checkpoint)

	
	# Create optimizer1
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	# optimizer = optim.SGD(model.parameters(),
	# 							   lr=args.lr,
	# 							   momentum=0.9,
	# 							   weight_decay=1e-4)
		





	# Model name
	
	model_name = args.model_name #+ str(args.lambda_)

	
	# Apply attention to subset of grid cells for training set
	grid_codes = deepcopy(original_grid_codes)

	
	# Train
	log.info('Training begins...')
	all_loss = []
	all_acc = []
	all_test_epochwise_acc = []
	test_loaders = [train_loader] + test_loaders
	# cells_include_test = [cells_include_train] + cells_include_test
	test_unique = [train_unique] + test_unique
	max_val_acc=0

	flag=0
	for epoch in range(1, args.epochs + 1):
		
		# grid_codes = deepcopy(original_grid_codes)

		# Training loop
	
		loss, acc = train(args, model, device, train_loader, grid_codes, analogy_levels_loc,optimizer, epoch)
		
		all_loss.append(loss)
		all_acc.append(acc)
		valacc = val(args,model, device, val_loader, grid_codes,analogy_levels_loc)
		if valacc > max_val_acc:
			print("Average validation accuracy increased from %s to %s"%(max_val_acc,valacc))
			max_val_acc = valacc
			# print("Saving model$$$$$")
			# save_nw(model,optimizer,epoch,'{}_run_{}_best.pth.tar'.format(args.model_name,args.run))
			# save_nw(model,optimizer,epoch,'{}_lambda_{}_run_{}_best.pth.tar'.format(args.model_name,args.lambda_,args.run))

			temp_test_acc=[]
			for t in range(len(test_loaders)):

				# Apply attention to subset of grid cells for training set
				# grid_codes = deepcopy(original_grid_codes)
				
				# Evaluate
				# print(t,cells_include_test[t])
			

				acc = test(args,t, model, device, test_loaders[t], grid_codes,analogy_levels_loc)
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
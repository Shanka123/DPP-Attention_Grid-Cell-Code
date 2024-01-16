
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
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


class Encoder(nn.Module):
	def __init__(self,in_dim):
		super(Encoder, self).__init__()

		
		self.encoder_N_units_per_FC_layer = 256
		self.N_latent = 256
		


		# self.conv1 = nn.Conv2d(1,8,kernel_size=3,stride=2, padding=0)
		
		self.fc1 = nn.Linear(in_dim, self.encoder_N_units_per_FC_layer)
		self.fc2 = nn.Linear(self.encoder_N_units_per_FC_layer, self.encoder_N_units_per_FC_layer)
		# self.dropout = nn.Dropout(0.5)
		self.linear_layer = nn.Linear(self.encoder_N_units_per_FC_layer, self.N_latent)
		
	   
		

	def forward(self, x):

	  
	  
		

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.linear_layer(x)
		
	 
	   
	   
	  
		return x
		


class LSTM(nn.Module):
	def __init__(self, in_dim):
		super(LSTM, self).__init__()
		log.info('Building LSTM and output layer...')
		self.in_dim = in_dim
		self.LSTM_size = 256
		
		self.lstm = nn.LSTMCell(in_dim, self.LSTM_size)
		self.score_out = nn.Linear(self.LSTM_size,1)
	def forward(self, A,B,C,D, device):
		
		batch_size = A.shape[0]
		A_tag = torch.tensor([1, 0, 0,0]).float().repeat(batch_size,1).to(device)
		A_tagged = torch.cat((A_tag,A),dim=1)

		B_tag = torch.tensor([0, 1, 0,0]).float().repeat(batch_size,1).to(device)
		B_tagged = torch.cat((B_tag,B),dim=1)


		C_tag = torch.tensor([0, 0, 1,0]).float().repeat(batch_size,1).to(device)
		C_tagged = torch.cat((C_tag,C),dim=1)

	

		
		D_tag = torch.tensor([0, 0, 0,1]).float().repeat(batch_size,1).to(device)
		D_tagged = torch.cat((D_tag,D),dim=1)

		

		# Concatenate inputs as timeseries
		# ABCD_lstm_inputs = torch.cat((A1_tagged.unsqueeze(0),A2_tagged.unsqueeze(0),A3_tagged.unsqueeze(0),A4_tagged.unsqueeze(0),A5_tagged.unsqueeze(0),A6_tagged.unsqueeze(0),A7_tagged.unsqueeze(0),A8_tagged.unsqueeze(0),A9_tagged.unsqueeze(0),D_tagged.unsqueeze(0)), dim=0)
		ABCD_lstm_inputs = torch.cat((A_tagged.unsqueeze(0),B_tagged.unsqueeze(0),C_tagged.unsqueeze(0),D_tagged.unsqueeze(0)), dim=0)
		
		# Initialize hidden state
		hidden = torch.zeros(ABCD_lstm_inputs.shape[1], self.LSTM_size) .cuda()
		cell_state = torch.zeros(ABCD_lstm_inputs.shape[1], self.LSTM_size) .cuda()
		# Apply LSTM
		output = []
		for i in range(ABCD_lstm_inputs.size()[0]):
			hidden, cell_state = self.lstm(ABCD_lstm_inputs[i], (hidden, cell_state))
			output.append(hidden)

		output = torch.stack(output, dim=0)
		# print(output.shape)
		# print(output.shape)
		# lstm_out, (hidden, cell_state) = self.lstm(ABCD_lstm_inputs, (hidden, cell_state))
		lstm_final_out = output[-1,:,:]
		# Output layer
		score = self.score_out(lstm_final_out)
		return score



# def encoder(img):

# 	# Hyperparameters
# 	# Convolutional encoder
# 	encoder_N_conv_layers = 4
# 	encoder_N_conv_feature_maps = 32
# 	encoder_conv_stride = 2
# 	encoder_conv_kernel_size = 4
# 	encoder_conv_kernel_sizes = (np.ones(encoder_N_conv_layers) * encoder_conv_kernel_size).astype(np.int)
# 	encoder_conv_stride_sizes = (np.ones(encoder_N_conv_layers) * encoder_conv_stride).astype(np.int)
# 	encoder_conv_N_channels = np.ones(encoder_N_conv_layers).astype(np.int) * encoder_N_conv_feature_maps
# 	# FC encoder
# 	encoder_N_FC_layers = 2
# 	encoder_N_units_per_FC_layer = 256
# 	encoder_FC_size = np.ones((encoder_N_FC_layers), dtype = np.int) * encoder_N_units_per_FC_layer
# 	# Latent space
# 	N_latent = 256

# 	# Rescale image
# 	log.info('Scale image between 0 and 1...')
# 	img_scaled = img / 255.0

# 	# Encoder
# 	log.info('Encoder...')
# 	log.info('Convolutional layers...')
# 	encoder_conv_out = conv2d(img_scaled, encoder_conv_kernel_sizes, encoder_conv_N_channels, encoder_conv_stride_sizes, 
# 		scope='encoder_conv', reuse=tf.AUTO_REUSE)
# 	encoder_conv_out_shape = encoder_conv_out.shape
# 	log.info('FC layers...')
# 	encoder_conv_out_flat = tf.layers.flatten(encoder_conv_out)
# 	encoder_conv_out_flat_shape = int(encoder_conv_out_flat.shape[1])
# 	encoder_FC_out, encoder_FC_w, encoder_FC_biases = mlp(encoder_conv_out_flat, encoder_FC_size, scope='encoder_FC', reuse=tf.AUTO_REUSE)

# 	# Latent representation
# 	log.info('Latent representation...')
# 	latent, latent_linear_w, latent_linear_biases = linear_layer(encoder_FC_out, N_latent, scope="latent", reuse=tf.AUTO_REUSE)

# 	return latent


# def scoring_model(A, B, C, D, layernorm=False):

	
# 	batch_size = A.shape[0]
# 	# Add tags to each element
# 	A_tag = torch.tensor([1, 0, 0,0]).to(float).repeat(batch_size,1)
# 	A_tagged = torch.cat((A_tag,A),dim=1)

# 	B_tag = torch.tensor([0, 1, 0,0]).to(float).repeat(batch_size,1)
# 	B_tagged = torch.cat((B_tag,B),dim=1)

# 	C_tag = torch.tensor([0, 0, 1,0]).to(float).repeat(batch_size,1)
# 	C_tagged = torch.cat((C_tag,C),dim=1)

# 	D_tag = torch.tensor([0, 0, 0,1]).to(float).repeat(batch_size,1)
# 	D_tagged = torch.cat((D_tag,D),dim=1)

# 	# A_tag = tf.tile(tf.expand_dims(tf.constant([1, 0, 0, 0], dtype=tf.float32), 0), [batch_size, 1])
# 	# A_tagged = tf.concat([A_tag, A], axis=1)
# 	# B_tag = tf.tile(tf.expand_dims(tf.constant([0, 1, 0, 0], dtype=tf.float32), 0), [batch_size, 1])
# 	# B_tagged = tf.concat([B_tag, B], axis=1)
# 	# C_tag = tf.tile(tf.expand_dims(tf.constant([0, 0, 1, 0], dtype=tf.float32), 0), [batch_size, 1])
# 	# C_tagged = tf.concat([C_tag, C], axis=1)
# 	# D_tag = tf.tile(tf.expand_dims(tf.constant([0, 0, 0, 1], dtype=tf.float32), 0), [batch_size, 1])
# 	# D_tagged = tf.concat([D_tag, D], axis=1)
# 	# Concatenate inputs as timeseries
# 	ABCD_lstm_inputs = torch.cat((A_tagged, B_tagged, C_tagged, D_tagged), dim=0)
# 	# LSTM
# 	# log.info("LSTM...")
# 	# if layernorm:
# 	# 	lstm_all_output = lstm_layernorm(ABCD_lstm_inputs, LSTM_size, scope="lstm", reuse=tf.AUTO_REUSE)
# 	# else:
# 	# lstm_all_output = lstm(ABCD_lstm_inputs, LSTM_size, scope="lstm", reuse=tf.AUTO_REUSE)
# 	# lstm_final_output = lstm_all_output[-1, :, :]
# 	# # Linear layer
# 	# log.info("Linear scoring layer...")
# 	# score, score_w, score_biases = linear_layer(lstm_final_output, 1, scope='score_linear', reuse=tf.AUTO_REUSE)

# 	# return score



# def encode_analogy_objs(imgs, ABCD, not_D):

# 	# Extract images for analogy terms (A, B, C, D) and foils (all objects != D)
# 	log.info('Extracting analogy terms...')
# 	A_img = extract_objects(imgs, ABCD[:, 0])
# 	B_img = extract_objects(imgs, ABCD[:, 1])
# 	C_img = extract_objects(imgs, ABCD[:, 2])
# 	D_img = extract_objects(imgs, ABCD[:, 3])
# 	not_D_imgs = extract_objects(imgs, not_D)

# 	# Get latent codes
# 	log.info('Building encoders...')
# 	log.info('A...')
# 	A_latent = encoder(A_img)
# 	log.info('B...')
# 	B_latent = encoder(B_img)
# 	log.info('C...')
# 	C_latent = encoder(C_img)
# 	log.info('D...')
# 	D_latent = encoder(D_img)
# 	log.info('Foils...')
# 	all_foil_latent = []
# 	N_foils = int(not_D.shape[1])
# 	for foil in range(N_foils):
# 		log.info('foil ' + str(foil+1) + '...')
# 		all_foil_latent.append(encoder(tf.gather(not_D_imgs, foil, axis=1)))
# 	all_foil_latent = tf.stack(all_foil_latent, axis=1)

# 	return A_latent, B_latent, C_latent, D_latent, all_foil_latent
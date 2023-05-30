import torch
# set seed value for reproducibility
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# define figure default dpi

plt.rcParams['figure.dpi'] = 500

class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self, arg):
		super(Encoder, self).__init__()
		self.lc1 = nn.Linear(arg.input_shape*arg.input_shape, arg.encoder)
		self.lc2 = nn.Linear(arg.encoder, arg.latent_dim)


	def forward(self, x):
		x = torch.flatten(x, start_dim=1)
		x = F.relu(self.lc1(x))
		x = self.lc2(x)
		return x


class Decoder(nn.Module):
	"""docstring for Decoder"""
	def __init__(self, arg):
		super(Decoder, self).__init__()
		self.input_shape = arg.input_shape
		self.lc1 = nn.Linear(arg.latent_dim, arg.decoder)
		self.lc2 = nn.Linear(arg.decoder, arg.input_shape*arg.input_shape)

	def forward(self, z):
		z = F.relu(self.lc1(z))
		z = torch.sigmoid(self.lc2(z))
		z = z.reshape(-1,1,self.input_shape, self.input_shape)

		return z


class Autoencoder(nn.Module):
	"""docstring for Autoencoder"""
	def __init__(self, arg):
		super(Autoencoder, self).__init__()
		self.encoder = Encoder(arg)
		self.decoder = Decoder(arg)

	def forward(self, x):
		z = self.encoder(x)
		x_prime = self.decoder(z)
		return x_prime



class Model(object):
	"""docstring for Model"""
	def __init__(self, arg):
		super(Model, self).__init__()
		self.lr = arg.learning_rate
		self.epochs = arg.epochs
		self.batch_size = arg.batch_size
		self.device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'
		self.autoencoder = Autoencoder(arg).to(self.device)

	def prepare_dataset(self):
		data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data',transform=torchvision.transforms.ToTensor(),
			download=True),batch_size=self.batch_size, shuffle=True)

		return data


	def train(self, data):

		opt = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
		for epoch in tqdm(range(self.epochs), colour="green"):
			t = 0
			for x, y in data:
				x = x.to(self.device)
				opt.zero_grad()
				x_hat = self.autoencoder(x)
				loss = ((x-x_hat)**2).sum()
				loss.backward()
				opt.step()
				t += 1
			print(f'T: {t}')


	def reconstruction(self):
		w, h = 28, 28
		n = 12
		r1 = (-5,10)
		r2 = (-10,10)

		img = np.zeros((n*w, n*h))
		for i,y in enumerate(np.linspace(*r1,n)):
			for j, x in enumerate(np.linspace(*r2,n)):
				z = torch.Tensor([[x,y]]).to(self.device)
				x_hat = self.autoencoder.decoder(z)
				x_hat = x_hat.reshape(w,h).to('cpu').detach().numpy()
				img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
		plt.imshow(img, extent=[*r1, *r2])
		plt.savefig("reconstruction.png")



		
		
		

from autoencoder import Autoencoder, Encoder, Decoder, Model
from arg_parser import ARGS

def main():
	model = Model(ARGS)
	data = model.prepare_dataset()
	model.train(data)
	model.reconstruction()


if __name__ == '__main__':
	main()
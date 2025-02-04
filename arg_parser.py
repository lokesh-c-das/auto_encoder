import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", default=2, type=int, help="Latent space dimension..")
parser.add_argument("--input_shape",default=28,type=int, help="Input Image size (nxn)..")
parser.add_argument("--encoder", default=16, type=int, help="Encoder hidden neuron units..")
parser.add_argument("--decoder", default=16,type=int, help="Decoder hidden neuron units..")
parser.add_argument("--batch_size",default=32, type=int, help="Mini batch Size..")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Model Learning Rate...")
parser.add_argument("--epochs",default=50, type=int, help="Epochs to train")
ARGS = parser.parse_args()
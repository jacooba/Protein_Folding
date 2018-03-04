import os


IN_WIDTH = 84
IN_HEIGHT = 84
IN_CHANNELS = 4

NUM_CONV_LAYERS = 3
CHANNEL_SIZES = [32, 64, 32]
CONV_KERNEL_SIZES = [(8, 8), (4, 4), (3, 3)]
CONV_STRIDES = [(4, 4), (2, 2), (1, 1)]
FC_INPUT_SIZE = 3872
FC_SIZE = 512

assert len(CHANNEL_SIZES) == len(CONV_KERNEL_SIZES) == len(CONV_STRIDES) == NUM_CONV_LAYERS

SAVE_FREQ = 500



NUM_ACTIONS = 4 #action "0" is up, "1" is right, etc
K = int("inf") #the k-step return

L = 40 #protein length
VEC_SZ = L-1 #the length of the input vector (HP string after first is placed)

NUM_ENVS = 5
NUM_EPOCHS = 4
GAMMA = 0.99

ENTROPY_REGULARIZATION_WEIGHT = 0.01


GRAPH_DIR = os.path.join(os.getcwd(), "Graph")
MODEL_DIR = os.path.join(os.getcwd(), "Model")
MODEL_PATH = os.path.join(MODEL_DIR, "model")



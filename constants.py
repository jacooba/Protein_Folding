import os


L = 40 #protein length
VEC_SZ = L-1 #the length of the input vector (HP string after first is placed)

IN_WIDTH = 2 * L
IN_HEIGHT = 2 * L
IN_CHANNELS = 2

SMALL_CONV_CHANNELS = [5]
SMALL_CONV_KERNELS = [3]
SMALL_CONV_STRIDES = [1]

SMALL_FLATTEN_SIZE = IN_WIDTH * IN_HEIGHT * SMALL_CONV_CHANNELS[-1] / np.prod(np.array(SMALL_CONV_STRIDES))

BIG_CONV_CHANNELS = [5, 10, 20, 30]
BIG_CONV_KERNELS = [int(math.sqrt(L / 2)), int(math.sqrt(L / 4)), 4, 2]
BIG_CONV_STRIDES = [1, 2, 3, 4]

BIG_FLATTEN_SIZE = IN_WIDTH * IN_HEIGHT * BIG_CONV_CHANNELS[-1]) / np.prod(np.array(BIG_CONV_STRIDES))

PADDING = "SAME"

FC_SIZES = [1024]

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

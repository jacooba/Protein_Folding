
import constants as c

from a2c import A2C
from collections import deque

import matplotlib.pyplot as plt
import sys, os, glob
import numpy as np
import argparse



class Runner:
    def __init__(self):
        #model
        self.model = Net()

    def train(self):
        print "\nWill TRAIN"
        self.run(train=True)

    def val(self):
        if not os.path.exists(c.MODEL_PATH):
            print "\n\nNo Model, validating untrained model\n\n"

        print "\nVALIDATING..."
        self.run(display=True)

    def run(self, train=False, display=False):
        protein_envs = np.random.choice([0, 1], size=[c.NUM_ENVS, c.L])
        protein0_scores = []
        for epoch in xrange(c.NUM_EPOCHS):
            print "ON EPOCH", epoch
            print "Acting in environments first..."
            all_states = []
            all_actions = []
            all_rds = [] #discounted reward to end (return)
            #COLECT DATA for all envs in batch
            batch = []
            for j, protein in enumerate(protein_envs):
                print "on protein", j
                states, actions, rds, total_r = act_in_env(protein, display=display)
                all_states.extend(states)
                all_actions.extend(actions)
                all_rds.extend(rds)
                print "protein", j, "final_score", total_r
                if j == 0:
                    protein0_scores.append(total_r)
            #TRAIN on batch 
            if train:
                print "training..."
                self.model.train_step(all_states, all_actions, all_rds)
        #plot scores over epochs
        if train:
            plot_scores(protein0_scores, "protein 0")

    def act_in_env(self, protein, display=False):
        #these will be returned
        states = []
        actions = []
        rds = [] #reward disctouted to end of episode (the Return)
        total_r = 0.0

        s = initial_state(protein)
        if display:
            display_state(s)
        # when theses queues hit size "k", we will beging popping form them:
        s_buffer = deque()
        a_buffer = deque()
        r_buffer = deque()
        # act in one env and collect data
        while s:
            a = self.model.get_actions([s], [get_valid_actions(s)])[0]
            r, s_next = T(s, a)

            s_buffer.appendleft(s)
            a_buffer.appendleft(a)
            r_buffer.appendleft(r)
            total_r += r

            s = s_next
            if display:
                display_state(s)

            #pop from buffer and calculate rds
            if len(s_buffer) >= c.K:
                states.append(s_buffer.pop())
                actions.append(a_buffer.pop())

                r_d = r_k(r_buffer)[0] + (c.GAMMA**c.K)*self.model.get_values([s_next])[0]
                rds.append(r_d)
                r_buffer.pop()

        #s was terminal
        #finish off buffer. next state is terminal.
        states.extend(s_buffer)
        actions.extend(a_buffer)
        rds.extend(r_k(r_buffer))


        return states, actions, rds, total_r

    #calculate the k-step look-ahead (discounted rewards) over a queue of rewards of length k
    @staticmethod
    def r_k(rewards):
        rks = deque()
        for r in reversed(rewards):
            future_reward = c.GAMMA*rks[0] if rks else 0.0
            rks.appendleft[r+future_reward]
        return rks

        # explicit version:
        # rks = [0 for _ in xrange(len(rewards))]
        # for i in reversed(xrange(len(rewards))):
        #     future_rk = c.GAMMA*rks[i+1] if i<len(rks) else 0.0
        #     rks[i] = rewards[i] + future_rk
        # return rks

    @staticmethod
    def plot_score(scores, line_label):
        plt.plot(range(c.NUM_EPOCHS), scores, label=line_label)
        plt.title("Score by Epoch")
        plt.xlabel('epoch')
        plt.ylabel('score')
        plt.legend()
        plt.show()

    @staticmethod
    def initial_state(protein): #returns (board_array, sequence_array, current_aminoacid)
        #set up board arrray (current board and one-hot board showing current locatoin)
        center_aminoacid = protein[0]

        board = np.zeros((2*c.L, 2*c.L))
        board[c.L/2, c.L/2] = center_aminoacid

        board_pos = np.zeros((2*c.L, 2*c.L))
        board_pos[c.L/2, c.L/2] = 1

        board_array = np.stack([board, board_pos], axis=-1)

        #set up sequence array
        sequence = protein[1:]
        sequence_pos = np.zeros_like(sequence)
        sequence_pos[0] = 1
        sequence_array = np.stack([sequence, sequence_pos], axis=0)

        #set up current aminoacid
        current_aminoacid = sequence[0]

        return (board_array, sequence_array, current_aminoacid)


    @staticmethod
    def get_valid_actions(state):
        board_array, _, _ = state
        board, _ = board_array
        r, c = get_curr_row_col(state)

        up_valid = int(board[r-1,c]==0)
        right_valid = int(board[r,c+1]==0)
        down_valid = int(board[r+1,c]==0)
        left_valid = int(board[r,c-1]==0)

        return np.array([up_valid, right_valid, down_valid, left_valid])

    @staticmethod
    def display_state(state):
        board_array, sequence_array, current_aminoacid = state
        #print board_array[0]
        sys.stdout.write("\r" + str(board_array[0]))
        sys.stdout.flush()

    @staticmethod
    #returns (reward, None) if terminal
    def T(state, action):
        board_array, sequence_array, current_aminoacid = state
        board, board_pos = board_array
        r, c = get_curr_row_col(state)
        new_r, new_c = get_new_row_col(r, c, action)

        ## reward ##
        reward = 0.0
        #consider all acids around new acid places
        for other_acid_r, other_acid_c in [(new_r-1,new_c), (new_r,new_c+1), (new_r+1,new_c), (new_r,new_c-1)]:
            if other_acid_r==r and other_acid_c==c: #the other acid is the one we came from 
                continue
            if current_aminoacid==board[other_acid_r,other_acid_c]: #acid you placed is same tyoe as one next to it
                reward += 1

        ## new_state ##
        #board
        new_board = np.copy(board)
        new_board_pos = np.zeros_like(board_pos)
        new_board_pos[new_r,new_c] = 1
        new_board[new_r,new_c] = current_aminoacid
        new_board_array = np.stack([board, board_pos], axis=-1)

        #sequence
        sequence, sequence_pos = sequence_array
        cur_index = np.nonzero(sequence_pos)[0]
        if cur_index == len(sequence_pos)-1: #at last postion. terminal next state.
            return reward, None
        new_sequence = np.copy(sequence)
        new_sequence[cur_index] = 0
        new_sequence_pos = np.zeros_like(sequence)
        new_sequence_pos[cur_index+1] = 1
        sequence_array = np.stack([new_sequence, new_sequence_pos], axis=0)

        #new current amino acid
        new_current_aminoacid = new_sequence[cur_index+1]

        #combined
        new_state = (new_board_array, new_sequence_array, new_current_aminoacid)


        return reward, new_state

    @staticmethod
    def get_curr_row_col(state):
        board_array, sequence_array, current_aminoacid = state
        board, board_pos = board_array
        row_wrapped, col_wrapped = np.nonzero(board_pos)
        r, c = row_wrapped[0], col_wrapped[0]
        return r, c

    @staticmethod
    def get_new_row_col(r, c, action):
        if action == 0:
            new_r, new_c = r-1, c
        if action == 1:
            new_r, new_c = r, c+1
        if action == 2:
            new_r, new_c = r+1, c
        if action == 3:
            new_r, new_c = r, c-1
        return new_r, new_c 

    # replaced by just keeping running sum of reward
    # @staticmethod
    # def score_state(state):
    #     pass



# helper functions #

def delete_model_files():
    modelFiles = glob.glob(os.path.join(c.MODEL_DIR, "*"))
    graphFiles = glob.glob(os.path.join(c.GRAPH_DIR, "*"))
    for f in modelFiles+graphFiles:
        os.remove(f)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--new',
                        help="Delete old model.",
                        dest='new',
                        action='store_true',
                        default=False)

    parser.add_argument('--val',
                        help="Validate model.",
                        dest='val',
                        action='store_true',
                        default=False)

    parser.add_argument('--train',
                        help="Train model.",
                        dest='train',
                        action='store_true',
                        default=False)

    return parser.parse_args()


# main #

def main():
    args = get_args()

    if args.new:
        #remove all model files
        delete_model_files()
    if args.train or args.val:
        runner = Runner()

    if args.train:
        runner.train()
    if args.val:
        runner.val()




if __name__ == "__main__":
    main()

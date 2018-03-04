
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
                states, actions, rds = act_in_env(protein, display=display)
                all_states.extend(states)
                all_actions.extend(actions)
                all_rds.extend(rds)
                score = score_final_state(states[-1])
                print "protein", j, "score", score
                if j == 0:
                    protein0_scores.append(score)
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

        s = initial_state(protein)
        if display:
            display_state(s)
        # when theses queues hit size "k", we will beging popping form them:
        s_buffer = deque()
        a_buffer = deque()
        r_buffer = deque()
        # act in one env and collect data
        while s:
            a = self.model.get_actions([s], get_valid_actions(s))[0]
            r, s_next = T(s, a)

            s_buffer.appendleft(s)
            a_buffer.appendleft(a)
            r_buffer.appendleft(r)

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


        return states, actions, rds

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
    def initial_state(protein):
        pass

    @staticmethod
    def get_valid_actions(state):
        pass

    @staticmethod
    #returns (reward, None) if terminal
    def T(state, action):
        pass

    @staticmethod
    def display_state(state):
        pass

    @staticmethod
    def score_final_state(state):
        pass



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

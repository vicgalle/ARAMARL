class KUHN():
    """
    A two-agent environment for a kuhn poker game.
    The "state" is s_t = (c_{d}, c_{a}) with c_{d}, c_{a} the cards of player 1
    and player 2, respectively.
    Player 1 (P1) actions are represented as vectors of 3 components.
    The firs is the action taken in the second round given that his opponent has
    passed. The second one is the action taken in the second round given that
    his opponent has bet. The third one is the action taken in the first round.
    Player 2 (P2) can either pass (0) or bet (1).
    """


    def __init__(self, max_steps, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.cards = np.array( [0,1,2] )
        #self.payout_mat = payouts
        #self.available_actions = [
        #    np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
        #    for _ in range(self.NUM_AGENTS)
        #]
        self.step_count = None


    #global deal
    def deal(self):
        return np.random.choice(self.cards, 2, replace = False)

    def reset(self):
        self.step_count = 0
        #init_state = np.zeros((self.batch_size, self.NUM_STATES))
        #init_state[:, -1] = 1
        #observations = [init_state, init_state]
        #info = [{'available_actions': aa} for aa in self.available_actions]
        observations = self.deal()
        return observations#, info


    def step(self, action, state):
        ac0, ac1 = action
        ac0 = np.array([ ac0[2], ac0[ac1] ])
        win = -2.0*np.argmax(state) + 1.0 #+1 is P1 has the biggest card, -1 otherwise

        self.step_count += 1

        # Zero-Sum so we just need one componet where we store reward for P1

        if ac0[0] == ac1:
            reward = win + win*ac1
        else:
            if ac0[0] == 1:
                reward = 1
            else:
                reward = (2*win +1)*ac0[1] - ac1

        rewards = [reward, -reward]
        observations = self.deal()

        """
        # The state is a OHE vector indicating [CC, CD, DC, DD, initial], (iff NUM_STATES = 5)
        state0 = np.zeros((self.batch_size, self.NUM_STATES))
        state1 = np.zeros((self.batch_size, self.NUM_STATES))
        for i, (a0, a1) in enumerate(zip(ac0, ac1)):  # iterates over batch dimension
            rewards.append([self.payout_mat[a1][a0], self.payout_mat[a0][a1]])
            state0[i, a0 * 2 + a1] = 1
            state1[i, a1 * 2 + a0] = 1
        """

        done = (self.step_count == self.max_steps)
        #info = [{'available_actions': aa} for aa in self.available_actions]

        return observations, rewards, done

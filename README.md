# Scratch_nn_comprehension
A repo made to understand neural networks from scratch + gymnasium use

## Authors 
Mathieu Suchet

## First try folder
This folder is my first initiation to agents with gymanisum environments. It contains the following content:
- Multi-worker agent learning (Unstable/Not working)
- Evironment rendering
- Agent with neural network
- Live data display (Not working)
- Listeners to display learning advancements

The agent here is learning through experience, it means it collects states during a certain number of episodes and then learn on it
### For developers/mathematicians here

**Everything stated beyond that point may be false, it represents what i concluded from my tests**

``` 
def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            for listener in self.replay_listeners:
                listener.on_replay_states_fail()
            return

        for listener in self.replay_listeners:
            listener.on_replay_start()

        batch = random.sample(self.memory, self.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward + self.GAMMA * np.amax(self.nn.predict(np.reshape(state_next, [1, self.state_space]))) * terminal
            q_values = self.nn.predict(np.reshape(state, [1, self.state_space]))
            q_values[0][action] = q_update
            self.nn.fit(np.reshape(state, [1, self.state_space]), q_values, verbose=0)
            self.steps += 1

        for listener in self.replay_listeners:
            listener.on_replay_end()
```

This is the function which represents the learning part of the agent

Here is how it works :
  - Pick a sample of states in the group of state harvested by the agent
- Calculate the "score" of the action for the state following the current state
  - I used a "GAMMA" constant equals to 0.99, this constant is here to represent how important the future rewards are. If this gamma is superior to 1, it can lead to infinite scores, so we have to pick a number between 0 and 1 (before 0 would transform good reward into bad reward and vice versa). The lower gamma is, the lower the rewards will be accounted for future states
  - ```np.amax(self.nn.predict(np.reshape(state_next, [1, self.state_space])))```

    This part represents the action that the network would have taken for the next state

  - If the state is a terminal one (end of state), all of this will give 0 (you'll understand the impact of such a decision later)

- Calculate what the network would have done for the current state and replace the chosen action probability by the previously calculated result (This result will not be used as a probability for an action to be picked, it'll be used to tell whether the network made a good or bad decision during the learning phase)
- Last step is to use the "fit" function, which changes the parameters of the network based on the state and the "q_values" represented by how good each action were for that state. If a number among the q_values is way bigger than the others, the network will understand that, given this state, the corresponding action has to be picked. Same goes for negative/very little q_value.
  



## Second try folder
This folder is my second initiation to agents with gymansium environments. It contains the following content:

- Level 1 : Basic environment rendering
  - Environment rendering
  - Verbose
  - Environment live-info
  - Agent live-info
  
- Level 2 : Dividing in modules
   - Environment rendering made in a different class
   - Agent actions made in a different class
   - Termination/Truncation check done in a different class

# Budget_Allocation_DRL

Advancements in budget allocation within portfolio management have been significantly influenced by the integration of artificial intelligence (AI) and machine learning techniques. A notable development is the application of Deep Reinforcement Learning that combines the strengths of reinforcement learning and deep learning, particularly well-suited for solving complex decision-making problems where an agent learns to achieve goals through trial and error, guided by feedback in the form of rewards.

![Image taken from Richard S. Sutton and Andrew G. Barto. Reinforcement Learning An Introduction.](img/rl_life_cycle.png)

By interacting with an environment like in Figure 1,  and making decisions through a policy 𝜋 that takes actions. The objective is to find the optimal policy 𝜋* that maximizes the future rewards.

It is well known that today's DeepRL algorithms are able to learn advanced strategies, achieved through algorithms such as Deep Q-learning, Policy Gradient, Actor-Critic models, together with exploration/exploitation trade-off. These strategies can also be transferred to industrial applications such as robotics, gaming, finance, healthcare, among others.
Specifically for the problem of budget allocation in investment portfolios, we define our Markov Decision Process (MDP) as follows:
States ($S_t$): The state at time t, captures the current market conditions and portfolio composition. This may include features such as asset prices, returns, volatility, macroeconomic indicators, and the current allocation of funds across assets. The signal features we used as states were 7-day and 30-day volatility, and the price trend at 3, 7, and 15 days.
Action (At): The action are percentages of how much money must be distributed in each of the assets.
Reward (Rt): The reward is a numerical value that reflects the performance of the portfolio after taking an action. In our case we are taking Sharpe ratio.
Discount factor 𝛄: for our case, we are taking it at 0.99.
Policy (𝜋): The policy is the strategy the agent uses to determine the best action given the current state. That is why our experiment rely on [2], where the author proposes to use a deterministic policy, based on an optimization problem, where the idea is to maximize the State-Value function, which is,

where Gt, represents the total discounted returns starting from time t.
Using this state-value function, we have that the problem to be optimized is as follows, 

where, 

To estimate the state-value function through a neural network, we do so following the similar estimation strategy for Deep Q-learning, where the cost function for the case of the Value function is

Readers should be aware that the state representations used in the Markov Decision Process (MDP) may be limited and could be enhanced with additional market information or features deemed relevant, and allows the model to detect market shifts. Furthermore, the neural network model can be improved by modifying its architecture to enhance performance and capture more signals from the input states, as long as you carefully monitor performance metrics like overfitting and sensitivity to hyperparameters.

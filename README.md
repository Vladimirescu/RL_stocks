# RL_stocks

Q-learning Agent that tries increase it's profit by buying/selling stocks from different companies.

Simple rule for buying : keep x% of your total current amount, and use (1-x)% to buy the stocks that the agent decides to (buys as much as he can).

Simple rule for selling: if the agent decides that the owned shares from a company should be sold (i.e. decides the action **sell** for stock **x**) it sells all owned shares from that company

**Example :**

Using the stock prices from Facebook, Starbucks and Apple we get the following **training** and **testing** data (which were obtained by taking the first, respectively last half of the full price list):


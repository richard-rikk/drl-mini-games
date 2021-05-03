#FrozenLake-v0 is considered "solved" when the agent obtains an average reward of at least 0.78 over 100 consecutive episodes.
#MountainCar-v0 is considered "solved" when the agent obtains an average reward of at least -110.0 over 100 consecutive episodes.
import games
import os

path = 'saved_models/'

if not os.path.exists(path):
    os.makedirs(path)

trainer = games.DQNTrainer(games.CAR_GAME)
trainer.train_model(2500)

#trainer = games.DQNTrainer(games.LAKE_GAME)
#trainer.train_model(30_000) #(1 hour)

#g = games.Game(games.LAKE_GAME, 1618899084, 1)
#print(f'Score: {g.evaulate(100)}')

#trainer = games.ACTrainer(games.LAKE_GAME)
#trainer.train_model(10) #(1:30 min)

#g = games.Gamev2(games.LAKE_GAME, 1619185164, subdir='acv2-668.tf')
#print(g.evaulate(episodes=100))

#trainer = games.ACv2Trainer(games.LAKE_GAME)
#trainer.train_model(1)




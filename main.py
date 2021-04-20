#FrozenLake-v0 is considered "solved" when the agent obtains an average reward of at least 0.78 over 100 consecutive episodes.
#MountainCar-v0 is considered "solved" when the agent obtains an average reward of at least -110.0 over 100 consecutive episodes.
import games
 


trainer = games.DQNTrainer(games.LAKE_GAME)
trainer.train_model(1000)

#trainer = games.DQNTrainer(games.LAKE_GAME)
#trainer.train_model(30_000) #(1 hour)

#g = games.Game(games.LAKE_GAME, 1618899084, 1)
#print(f'Score: {g.evaulate(100)}')

#trainer = games.ACTrainer(games.LAKE_GAME)
#trainer.train_model(100) #(1:30 min)

#g = games.Game(games.LAKE_GAME, 1618852313, modelType=1)
#g.play(steps=25)



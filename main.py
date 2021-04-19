import games

#trainer = games.ACTrainer(games.LAKE_GAME)
#trainer.train_model(2000)

#trainer = games.DQNTrainer(games.CAR_GAME)
#trainer.train_model(2000)

#g = games.Game(games.LAKE_GAME, 1618832600)
#g.play()

g = games.Game(games.LAKE_GAME, 1618835756, modelType=1)
g.play(steps=100)



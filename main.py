import games

trainer = games.DQNTrainer(games.LAKE_GAME)
trainer.train_model(30_000) #(1 hour)

#g = games.Game(games.LAKE_GAME, 1618837681)
#g.play()

#trainer = games.ACTrainer(games.LAKE_GAME)
#trainer.train_model(30_000) #(1:30 min)

#g = games.Game(games.LAKE_GAME, 1618852313, modelType=1)
#g.play(steps=25)



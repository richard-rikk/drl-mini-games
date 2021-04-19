from games.games import Game
import games

trainer = games.DQNTrainer(games.LAKE_GAME)
trainer.train_model(2000)

#g = games.Game(games.LAKE_GAME, 1618778066)
#g.play()

#g = games.Game(games.CAR_GAME, 1618823840)
#g.play(steps=10_000)

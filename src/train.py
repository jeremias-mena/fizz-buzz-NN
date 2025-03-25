import pickle
import random
from fizz_buzz import FizzBuzzNN, Util, Config

num_hidden = 30

x = [Util().binary_encoder(i) for i in range(101, 1024)]
y = [Util().f_buzz_encoder(i) for i in range(101, 1024)]

fb_network = FizzBuzzNN(neural_network=[
                                        [[random.random() for _ in range(10 + 1)] for _ in range(num_hidden)],
                                        [[random.random() for _ in range(num_hidden + 1)] for _ in range(4)]
                                        ])
fb_network.train(x, y)

with open(Config.train_model_path, "wb") as model_file:
    pickle.dump(fb_network, model_file)



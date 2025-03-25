from pathlib import Path

class Config:
    dir_train_model = Path(__file__).resolve().parent/'model'
    train_model_path = dir_train_model/'fizz_buzz_NN.pkl'
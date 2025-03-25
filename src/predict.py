import pickle
from fizz_buzz import Config, Util
from colorama import Fore, Style

with open(Config.train_model_path, "rb") as model_file:
    fb_network = pickle.load(model_file)

def predict() -> int:
    num_correct = 0
    for i in range(1, 101):
        x = Util().binary_encoder(i)
        predicted = Util().pos_max(fb_network.feed_forward(x)[-1])
        real = Util().pos_max(Util().f_buzz_encoder(i))
        labels = [str(i), "fizz", "buzz", "fizzbuzz"]
        if (predicted == real):
            print(f"Número: {i} |", f"Predicción: {labels[predicted]} |", f"Valor real: {labels[real]}")
            num_correct += 1
        else:
            print(f"Número: {i} |", Fore.RED + f"Predicción: {labels[predicted]} |", f"Valor real: {labels[real]}" + Style.RESET_ALL)
    return num_correct

if __name__ == "__main__":
    result = predict()
    print(f"Resultados correctos: {result}/100")
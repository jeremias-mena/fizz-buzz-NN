import math
from typing import List

class Util:
    def __init__(self):
        pass
    
    def sigmoid_func(self, x:float) -> float:
        y = 1 / (1 + math.exp(-x))
        return y
    
    def pos_max(self, x:List) -> int:
        pos = max(range(len(x)), key=lambda x_i: x[x_i])
        return pos
    
    def dot_product(self, x:List[float], y:List[float]) -> float:
        if len(x) == len(y):
            result =  sum(x_i * y_i for x_i, y_i in zip(x, y))
        else:
            raise(NameError(f"Ambos vectores deben tener la misma longitud: len(x) = {len(x)} y len(y) = {len(y)}"))
        return result
    
    def squared_dist(self, x:List[float], y:List[float]) -> float:
        if len(x) == len(y):
            substract_vec =  [x_i - y_i for x_i, y_i in zip(x, y)]
            result = self.dot_product(substract_vec, substract_vec)
        else:
            raise(NameError(f"Ambos vectores deben tener la misma longitud: len(x) = {len(x)} y len(y) = {len(y)}"))
        return result
    
    def gradient_step(self, x:List[float], gradient:List[float], step_size:float) -> List[float]:
        if len(x) == len(gradient):
           step = [step_size * x_i for x_i in x]
           result =  [x_i + s_i for x_i, s_i in zip(x, step)]
        else:
            raise(NameError(f"Ambos vectores deben tener la misma longitud: len(x) = {len(x)} y len(gradient) = {len(gradient)}"))
        return result
    
    def binary_encoder(self, x:int) -> List[float]:
        encoder: List[float] = []
        for i in range(10):
            encoder.append(x % 2)
            x = x // 2
        return encoder

    def f_buzz_enconcer(self, x: int) -> List[float]:
        if x % 15 == 0:
            return [0, 0, 0, 1]
        elif x % 5 == 0:
            return [0, 0, 1, 0]
        elif x % 3 == 0:
            return [0, 1, 0, 0]
        else:
            return [1, 0, 0, 0]
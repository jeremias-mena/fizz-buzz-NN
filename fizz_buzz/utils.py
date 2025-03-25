import math
from typing import List

class Util:
    def __init__(self):
        pass
    
    def sigmoid_func(self, x:float) -> float:
        y = 1 / (1 + math.exp(-x))
        return y
    
    def dot_product(self, x:List[float], y:List[float]) -> float:
        if len(x) == len(y):
            result =  sum(x_i * y_i for x_i, y_i in zip(x, y))
        else:
            raise(NameError(f"Ambos vectores deben tener la misma longitud: len(x) = {len(x)} y len(y) = {len(y)}"))
        return result
    
from fizz_buzz import Util

# tests for sigmoid_func 
assert round(Util().sigmoid_func(1), 2) == 0.73, False
assert round(Util().sigmoid_func(2), 2) == 0.88, False
assert round(Util().sigmoid_func(3), 2) == 0.95, False

# tests for pos_max
assert Util().pos_max([0, 1, 2, 3]) == 3, False
assert Util().pos_max([0, 3, 4, 1]) == 2, False
assert Util().pos_max([1, 5, 3, 2]) == 1, False

# tests for dot_product
assert Util().dot_product([0, 1, 2], [3, 4, 5]) == 14, False
assert Util().dot_product([3, 2, 5], [1, 2, 3]) == 22, False
assert Util().dot_product([1, 1, 1], [2, 2, 2]) == 6, False

# tests for squared_dist
assert Util().squared_dist([0, 1, 2], [3, 4, 5]) == 27, False
assert Util().squared_dist([3, 2, 5], [1, 2, 3]) == 8, False
assert Util().squared_dist([1, 1, 1], [2, 2, 2]) == 3, False

# tests for gradient_step
assert Util().gradient_step([1.0, 2.0, 3.0], [-0.1, -0.2, -0.3], 0.5) == [0.95, 1.9, 2.85], False
assert Util().gradient_step([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 0.5) == [1.05, 2.1, 3.15], False

# tests for binary_encoder
assert Util().binary_encoder(0) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], False
assert Util().binary_encoder(1) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], False
assert Util().binary_encoder(2) == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], False
assert Util().binary_encoder(3) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], False

# tests for f_buzz_encoder
assert Util().f_buzz_encoder(3) == [0, 1, 0, 0], False
assert Util().f_buzz_encoder(5) == [0, 0, 1, 0], False
assert Util().f_buzz_encoder(15) == [0, 0, 0, 1], False
assert Util().f_buzz_encoder(4) == [1, 0, 0, 0], False
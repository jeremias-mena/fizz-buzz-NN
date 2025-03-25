from typing import List
import tqdm
from fizz_buzz import Util

class FizzBuzzNN():
    def __init__(self, neural_network:List[List[List[float]]]):
        self.neural_network = neural_network
        pass
    
    def neuron_output(self, weights_w_bias:List[float], inputs:List[float]) -> float:
        result = Util().sigmoid_func(Util().dot_product(weights_w_bias, inputs))
        return result

    def feed_forward(self, input_vec:List[float]) -> List[List[float]]:
        outputs: List[List[float]] = []
        bias = [1]
        for layer in self.neural_network:
            input_w_bias = input_vec + bias
            output = [self.neuron_output(neuron, input_w_bias) for neuron in layer]
            outputs.append(output)
            input_vec = output
        return outputs

    def square_error_gradients(self, input_vec:List[float], target_vec:List[float]) -> List[List[List[float]]]:
        bias = [1]
        hidden_outputs, outputs = self.feed_forward(self.neural_network, input_vec)
        output_deltas = [output * (1 - output) * (output - target)
                         for output, target in zip(outputs, target_vec)]
        output_gradients = [[output_deltas[i] * hidden_output
                             for hidden_output in hidden_outputs + bias]
                             for i, output_neuron in enumerate(self.neural_network[-1])]
        hidden_deltas = [hidden_output * (1 - hidden_output) * Util().dot_product(output_deltas, [n[i] for n in self.neural_network[-1]])
                         for i, hidden_output in enumerate(hidden_outputs)]
        hidden_gradients = [[hidden_deltas[i] * input for input in input_vec + bias]
                            for i, hidden_neuron in enumerate(self.neural_network[0])]
        return [hidden_gradients, output_gradients]
    
    def train(self, x:List, y: List) -> None:
        learning_rate = 1.0
        with tqdm.trange(500) as t:
            for epoch in t:
                epoch_loss = 0.0
                for x_i, y_i in zip(x, y):
                    predicted = self.feed_forward(x)[-1]
                    epoch_loss += Util().squared_dist(predicted, y_i)
                    gradients = self.square_error_gradients(x_i, y_i)
                    self.neural_network = [[Util().gradient_step(neuron, gradient, -learning_rate)
                                            for neuron, gradient in zip(layer, layer_gradient)]
                                            for layer, layer_gradient in zip(self.neural_network, gradients)]
                t.set_description(f"fizz_buzz_loss: (loss: {round(epoch_loss, 2)}) ")
    
    


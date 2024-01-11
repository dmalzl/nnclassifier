from torch import nn
import itertools as it


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def generate_layers(n_input_features, n_output_features, hidden_layer_sizes):
    # generates layers in the given sizes
    input_layer = nn.Linear(n_input_features, hidden_layer_sizes[0])
    output_layer = nn.Linear(hidden_layer_sizes[-1], n_output_features)
    if len(hidden_layer_sizes) == 1:
        return [input_layer, output_layer]

    layers = [input_layer]
    for in_size, out_size in pairwise(hidden_layer_sizes):
        layers.append(
            nn.Linear(in_size, out_size)
        )

    layers.append(output_layer)

    for layer in layers:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        
    return layers


def interleave_activation(layers, activation):
    # interleaves layers with an activation function if given
    if not activation:
        return layers
        
    activation_interleaved_layers = []
    for layer in layers:
        activation_interleaved_layers.extend(
            [layer, activation()]
        )

    #activation_interleaved_layers.append(layers[-1])
    return activation_interleaved_layers


def build_network(
    n_input_features, 
    n_output_features, 
    hidden_layer_sizes, 
    activation = None
):
    # generates specified layers interleaves them with activation if given and returns a sequential network
    layers = generate_layers(
        n_input_features,
        n_output_features, 
        hidden_layer_sizes
    )
    activation_interleaved_layers = interleave_activation(
        layers,
        activation
    )

    return nn.Sequential(*activation_interleaved_layers)


class NeuralNetClassifier(nn.Module):
    # Class to use with torch ecosystem to train classifier
    def __init__(
        self, 
        n_input_features = 12, 
        n_output_features = 3, 
        hidden_layer_sizes = [5],
        activation = None
    ):
        super().__init__()
        self.network = build_network(
            n_input_features,
            n_output_features,
            hidden_layer_sizes,
            activation
        )
    
    def forward(self, X):
        return self.network(X)
    
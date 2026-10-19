import numpy as np
import torch
from torch import nn

INIT_METHOD_CHOICES = (
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
)
BENCHMARK_G_INIT_METHOD_CHOICES = (
    "scaled_static_inverse",
    "fluid_inverse",
)
G_INIT_METHOD_CHOICES = INIT_METHOD_CHOICES + BENCHMARK_G_INIT_METHOD_CHOICES


def initialization_metadata(init_method):
    """Return the documented torch initialization convention for a method."""
    method = _resolve_init_method(init_method)
    if method == "xavier_uniform":
        return {
            "method": method,
            "distribution": "uniform",
            "torch_api": "torch.nn.init.xavier_uniform_",
            "gain": 1.0,
            "bias_rule": "preserve torch.nn.Linear reset_parameters bias",
        }
    if method == "xavier_normal":
        return {
            "method": method,
            "distribution": "normal",
            "torch_api": "torch.nn.init.xavier_normal_",
            "gain": 1.0,
            "bias_rule": "preserve torch.nn.Linear reset_parameters bias",
        }
    if method == "kaiming_uniform":
        return {
            "method": method,
            "distribution": "uniform",
            "torch_api": "torch.nn.init.kaiming_uniform_",
            "mode": "fan_in",
            "nonlinearity": "relu",
            "a": 0.0,
            "gain": float(nn.init.calculate_gain("relu")),
            "gelu_note": (
                "GELU has no dedicated torch.nn.init gain; the implementation uses the "
                "ReLU gain approximation and records it explicitly."
            ),
            "bias_rule": "preserve torch.nn.Linear reset_parameters bias",
        }
    if method == "kaiming_normal":
        return {
            "method": method,
            "distribution": "normal",
            "torch_api": "torch.nn.init.kaiming_normal_",
            "mode": "fan_in",
            "nonlinearity": "relu",
            "a": 0.0,
            "gain": float(nn.init.calculate_gain("relu")),
            "gelu_note": (
                "GELU has no dedicated torch.nn.init gain; the implementation uses the "
                "ReLU gain approximation and records it explicitly."
            ),
            "bias_rule": "preserve torch.nn.Linear reset_parameters bias",
        }
    if method == "orthogonal":
        return {
            "method": method,
            "distribution": "orthogonal",
            "torch_api": "torch.nn.init.orthogonal_",
            "gain": 1.0,
            "bias_rule": "preserve torch.nn.Linear reset_parameters bias",
        }
    raise ValueError(f"Unsupported init_method={init_method!r}")


def _activation_and_default_init(activation, slope):
    if activation == 'ReLU':
        return nn.ReLU(), "kaiming_uniform"
    if activation == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=slope), "kaiming_uniform"
    if activation == "GELU":
        return nn.GELU(), "xavier_uniform"
    if activation == "SELU":
        return nn.SELU(), "xavier_uniform"
    if activation == "SiLU":
        return nn.SiLU(), "xavier_uniform"
    if activation == "CELU":
        return nn.CELU(alpha=slope), "xavier_uniform"
    raise Exception(
        f"{activation} not an available activation. "
        "Available options: ReLU, LeakyReLU, GELU, SELU, SiLU, CELU"
    )


def _resolve_init_method(init_method):
    if init_method is None:
        return None
    name = str(init_method)
    if name not in INIT_METHOD_CHOICES:
        raise ValueError(
            f"init_method={name!r} is not available. "
            f"Available options: {', '.join(INIT_METHOD_CHOICES)}"
        )
    return name


def _apply_weight_init(layer, init_method):
    if init_method == "xavier_uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif init_method == "xavier_normal":
        nn.init.xavier_normal_(layer.weight)
    elif init_method == "kaiming_uniform":
        nn.init.kaiming_uniform_(
            layer.weight,
            a=0.0,
            mode="fan_in",
            nonlinearity="relu",
        )
    elif init_method == "kaiming_normal":
        nn.init.kaiming_normal_(
            layer.weight,
            a=0.0,
            mode="fan_in",
            nonlinearity="relu",
        )
    elif init_method == "orthogonal":
        nn.init.orthogonal_(layer.weight)
    else:  # pragma: no cover - guarded by _resolve_init_method
        raise ValueError(f"Unsupported init_method={init_method!r}")


class Y_Net(nn.Module):
    
    def __init__(self, dim, units, num_layers, activation, slope, init_method=None):
        super(Y_Net, self).__init__()
        self.init_method = _resolve_init_method(init_method)

        self.linearlayers = nn.ModuleList([nn.Linear(dim, units)])

        for _ in range(num_layers - 1):  # num_layers is the number of hidden layers
            self.linearlayers.append(nn.Linear(units, units))
        
        self.linearlayers.append(nn.Linear(units, 1, bias=False))

        self.activation, default_init = _activation_and_default_init(activation, slope)
        if self.init_method is None:
            self.init_method = default_init
        
        for layer in self.linearlayers:
            if isinstance(layer, nn.Linear):
                _apply_weight_init(layer, self.init_method)

    
    def forward(self, x):

        # hidden layers
        for layer in self.linearlayers[:-1]:
            x = self.activation(layer(x))

        # pass through the output layer
        x = self.linearlayers[-1](x)

        return x



class GradDiffNet(nn.Module):
    
    def __init__(
        self,
        input_dim,
        output_dim,
        units,
        num_layers,
        activation,
        slope,
        init_method=None,
        output_bias=False,
    ):
        super(GradDiffNet, self).__init__()
        self.init_method = _resolve_init_method(init_method)
        self.output_bias = bool(output_bias)

        self.linearlayers = nn.ModuleList([nn.Linear(input_dim, units)])
        
        for _ in range(num_layers - 1):
            self.linearlayers.append(nn.Linear(units, units))
        
        self.linearlayers.append(nn.Linear(units, output_dim, bias=self.output_bias))

        self.activation, default_init = _activation_and_default_init(activation, slope)
        if self.init_method is None:
            self.init_method = default_init
        
        for layer in self.linearlayers:
            if isinstance(layer, nn.Linear):
                _apply_weight_init(layer, self.init_method)

    
    def forward(self, x):

        # hidden layers
        for layer in self.linearlayers[:-1]:
            x = self.activation(layer(x))

        # pass through the output layer
        x = self.linearlayers[-1](x)

        return x


class QGradNet(GradDiffNet):
    """Scalar network for the physical inventory derivative partial_q V."""

    def __init__(self, input_dim, units, num_layers, activation, slope, init_method=None):
        super().__init__(input_dim, 1, units, num_layers, activation, slope, init_method)


class Z_Net(GradDiffNet):
    """Backward-compatible alias for the control-relevant gradient-difference network."""

    def __init__(self, input_dim, *args):
        if len(args) == 4:
            output_dim = input_dim
            units, num_layers, activation, slope = args
            init_method = None
        elif len(args) == 5:
            if isinstance(args[2], str):
                output_dim = input_dim
                units, num_layers, activation, slope, init_method = args
            else:
                output_dim, units, num_layers, activation, slope = args
                init_method = None
        elif len(args) == 6:
            output_dim, units, num_layers, activation, slope, init_method = args
        else:
            raise TypeError(
                "Z_Net expects either (input_dim, units, num_layers, activation, slope) "
                "or (input_dim, output_dim, units, num_layers, activation, slope), "
                "with an optional init_method suffix."
            )

        super().__init__(input_dim, output_dim, units, num_layers, activation, slope, init_method)

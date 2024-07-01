import torch
import torch.nn.functional as F
import torch.nn as nn

polynomial = 'chebyshev'
polynomial_order = 3


# RESTRICTED TO -1 <= x <= 1
def compute_chebyshev_polynomials(x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        chebyshev_polys = [P0, P1]
        
        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            #Pn = ((2.0 * n + 1.0) * x * chebyshev_polys[-1] - n * chebyshev_polys[-2]) / (n + 1.0)
            Pn = 2 * x * chebyshev_polys[-1] -  chebyshev_polys[-2]
            chebyshev_polys.append(Pn)
        
        return torch.stack(chebyshev_polys, dim=-1)

# RESTRICTED TO -1 <= x <= 1
def compute_legendre_polynomials(x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]
        
        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)
        
        return torch.stack(legendre_polys, dim=-1)


# NOT RESTRICTED TO -1 <= x <= 1
def compute_hermite_polynomials(x, order):
    # Base case polynomials P0 and P1
    P0 = x.new_ones(x.shape)  # P0 = 1 for all x
    if order == 0:
        return P0.unsqueeze(-1)
    P1 = x  # P1 = x
    hermite_polys = [P0, P1]
    
    # Compute higher order polynomials using recurrence
    for n in range(1, order):
        Pn = 2* x * hermite_polys[-1] - (2* n * hermite_polys[-2])
        hermite_polys.append(Pn)
    
    return torch.stack(hermite_polys, dim=-1)


# ONLY FOR x >= 0
def compute_laguerre_polynomials(x, order):
    # Base case polynomials P0 and P1
    P0 = x.new_ones(x.shape)  # P0 = 1 for all x
    if order == 0:
        return P0.unsqueeze(-1)
    P1 = 1 - x  # P1 = 1 - x
    laguerre_polys = [P0, P1]
    
    # Compute higher order polynomials using recurrence
    for n in range(1, order):
        Pn = ((2.0 * n + 1.0 - x) * laguerre_polys[-1] - n * laguerre_polys[-2]) / (n + 1.0)
        laguerre_polys.append(Pn)
    
    return torch.stack(laguerre_polys, dim=-1)

# Get the lookup table for discrete Chebyshev polynomials around [-1, 1]
def get_lookup_table(order, steps, polynomial):
# Define the range of x values and the order of the polynomials
    x_values = torch.linspace(-1, 1, steps)  # Replace with your actual range of x values

    # Initialize the lookup table
    lookup_table = torch.zeros(len(x_values), order + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lookup_table = lookup_table.to(device)
    for i, x in enumerate(x_values):
        if polynomial == 'chebyshev':
            lookup_table[i] = compute_chebyshev_polynomials(x, order)  # Replace with your actual function
        elif polynomial == 'legendre':
            lookup_table[i] = compute_legendre_polynomials(x, order)
        elif polynomial == 'hermite':
            lookup_table[i] = compute_hermite_polynomials(x, order)
        elif polynomial == 'laguerre':
            lookup_table[i] = compute_laguerre_polynomials(x, order)
        else:
            raise ValueError('Polynomial not supported')
    return lookup_table 

def get_polynomials(x, lookup_table):
    # Get the position in the lookup table for the given x value
    indices = ((x - (-1)) / (1 - (-1)) * (len(lookup_table) - 1)).long()
    # Get the Chebyshev polynomials for the given x value
    return lookup_table[indices]


lookup_table = get_lookup_table(polynomial_order, 20000, polynomial)

class KAN_layer(nn.Module):  # Kolmogorov Arnold Network Layer with different polynomials according to lookup_table (KAN)
    def __init__(self, in_features, out_features, polynomial, polynomial_order=polynomial_order, base_activation=nn.SiLU, lookup_table=lookup_table, dropout_p=0.2):
        super(KAN_layer, self).__init__()  # Initialize the parent nn.Module class
        
        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.in_features = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.base_activation = base_activation()
        self.lookup_table = lookup_table
        self.dropout = nn.Dropout(dropout_p)
        self.polynomial = polynomial
        # ParameterList for the base weights of each layer
        self.base_weights = nn.ParameterList()
        # ParameterList for the polynomial weights for polynomial expansion
        self.poly_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        # Base weight for linear transformation in each layer
        self.base_weights.append(nn.Parameter(torch.randn(self.out_features, self.in_features)))
        
        
        # Polynomial weight for handling polynomial expansions
        self.poly_weights.append(nn.Parameter(torch.randn(self.out_features, self.in_features * (polynomial_order + 1))))

        
        # Layer normalization to stabilize learning and outputs
        self.layer_norms.append(nn.LayerNorm(self.out_features))

        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def forward(self, x):
            # Ensure x is on the right device from the start, matching the model parameters
            x = x.to(self.base_weights[0].device)

            for i, (base_weight, poly_weight, layer_norm) in enumerate(zip(self.base_weights, self.poly_weights, self.layer_norms)):
                # Apply base activation to input and then linear transform with base weights
                base_output = F.linear(self.base_activation(x), base_weight)
                if self.polynomial in ['chebyshev', 'legendre', 'hermite']:
                    x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
                if self.polynomial == 'laguerre':
                    x_normalized = (x / x.min()) / (x.max()-x.min())
                polynomial_basis = get_polynomials(x_normalized, self.lookup_table)
                polynomial_basis = polynomial_basis.view(x.size(0), -1)

                # Compute polynomial output using polynomial weights
                poly_output = F.linear(polynomial_basis, poly_weight)
                # Combine base and polynomial outputs, normalize, and activate
                x = self.dropout(self.base_activation(layer_norm(base_output + poly_output)))

            return x


class MLKAN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, polynomial_order=3, base_activation=nn.SiLU, lookup_table=lookup_table):
        super(MLKAN, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                self.layers.append(KAN_layer(layer_sizes[i], layer_sizes[i+1], polynomial, polynomial_order, base_activation, lookup_table, dropout_p=0))
            else:
                self.layers.append(KAN_layer(layer_sizes[i], layer_sizes[i+1], polynomial, polynomial_order, base_activation, lookup_table, dropout_p=0.2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# KAN forma las capas de parámetros entre las dimensiones que transformamos, por lo que 
# hay una KAN de 768 a 256, otra de 256 a 128, otra de 128 a 64 y otra de 64 a 4.
""" model = MLKAN(768, [256, 128, 64], 4)
print(model) """

""" print('parameters:')
for name, param in (model.named_parameters()):
    if param.requires_grad:
        print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
    else:
        print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))

num_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)
print('\ttotal:', num_params) """

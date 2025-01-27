import torch
import torch.nn.functional as F
import torch.nn as nn
from utils_gegenbauer import *


# RESTRICTED TO -1 <= x <= 1
def compute_chebyshev_polynomials_1(x, order):
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

def compute_chebyshev_polynomials_2(x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = 2*x  
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


def compute_gegenbauer_polynomials(x, order, alpha, activation, bias=None):
    # x: [batch_size, hidden_size]
    # alpha: [hidden_size, hidden_size]

    P0 = x.new_ones(x.shape)  # P0 = 1
    if order == 0:
        return P0.unsqueeze(-1)

    alpha = activation(alpha)  # Activación sobre alpha

    if bias is not None:
        P1 = 2 * F.linear(x, alpha, bias)  # lambda_x = x @ alpha^T + bias
    else:
        P1 =  2 * F.linear(x, alpha)        # lambda_x = x @ alpha^T

    gegenbauer_polys = [P0, P1]

    for n in range(1, order):
        # Primer término: 2(n x + lambda^T x) * Cn
        first_term = 2 * (n*x + F.linear(x, alpha)) * gegenbauer_polys[-1]  # Element-wise multiplication

        
        # Segundo término: (n I + 2 λ^T - I) * C_{n-1}
        
        # Creamos la matriz (n - 1) * I
        n_minus_one_I = (n - 1) * torch.eye(x.size(1), device=x.device)  # [hidden_size, hidden_size]

        # Sumamos 2 λ^T
        second_term_matrix = n_minus_one_I + 2 * alpha  # [hidden_size, hidden_size]

        # Multiplicamos C_{n-1} por la matriz resultante
        second_term = torch.matmul(gegenbauer_polys[-2], second_term_matrix)  # [batch_size, hidden_size]

        # Calculamos Pn
        Pn = (first_term - second_term) / (n + 1)

        gegenbauer_polys.append(Pn)

    return torch.stack(gegenbauer_polys, dim=-1)


class KAN(nn.Module):
    def __init__(self, args, in_features, out_features, polynomial, polynomial_order=3, dropout_p=0.2):
        super(KAN, self).__init__()

    # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.in_features = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.base_activation = nn.SiLU()

        
        self.dropout = nn.Dropout(dropout_p)
        self.polynomial = polynomial

        # ParameterList for the polynomial weights for polynomial expansion
        self.poly_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()
        
        # Polynomial weight for handling polynomial expansions
        self.poly_weights.append(nn.Parameter(torch.randn(self.out_features, self.in_features * (polynomial_order + 1))))
        
        for weight in self.poly_weights:
            nn.init.xavier_uniform_(weight, nonlinearity='linear')   # linear
            #nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # linear
        
        # Last normalization used before activation
        if args['final_normalization'] == 'custom_norm':
            self.final_norm = CustomNorm(self.out_features)
        elif args['final_normalization'] == 'z_score_norm':
            self.final_norm = z_score_norm
        elif args['final_normalization'] == 'z_score_norm_2d':
            self.final_norm = z_score_norm_2d

    def forward(self, x):
            # Ensure x is on the right device from the start, matching the model parameters
            x = x.to(self.poly_weights[0].device)
            for i, (poly_weight) in enumerate(self.poly_weights):
                x_normalized = 2 * (x - x.min()) / (x.max() - x.min() +1e-8) - 1
                if self.polynomial == 'Chebyshev_1':
                    polynomial_basis = compute_chebyshev_polynomials_1(x_normalized, self.polynomial_order)
                elif self.polynomial == 'Chebyshev_2':
                    polynomial_basis = compute_chebyshev_polynomials_2(x_normalized, self.polynomial_order)
                elif self.polynomial == 'Legendre':
                    polynomial_basis = compute_legendre_polynomials(x_normalized, self.polynomial_order)
                else:
                    raise ValueError('Polynomial not implemented')
                polynomial_basis = polynomial_basis.view(x.size(0), -1)

                # Compute polynomial output using polynomial weights
                poly_output = F.linear(polynomial_basis, poly_weight)
                # Combine base and polynomial outputs, normalize, and activate
                x = self.dropout(self.base_activation(self.final_norm(poly_output)))
            return x


class KAN_layer(nn.Module):  # Kolmogorov Arnold Network Layer with different polynomials according to lookup_table (KAN)
    def __init__(self, args, in_features, out_features, polynomial, lookup_table, polynomial_order=3, dropout_p=0.2):
        super(KAN_layer, self).__init__()  # Initialize the parent nn.Module class
        
        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.in_features = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.base_activation = nn.SiLU()

        self.lookup_table = lookup_table
        self.dropout = nn.Dropout(dropout_p)
        self.polynomial = polynomial
        
        # ParameterList for the base weights of each layer
        self.base_weights = nn.ParameterList()
        # ParameterList for the polynomial weights for polynomial expansion
        self.poly_weights = nn.ParameterList()

        # Base weight for linear transformation in each layer
        self.base_weights.append(nn.Parameter(torch.randn(self.out_features, self.in_features)))
                
        # Polynomial weight for handling polynomial expansions
        self.poly_weights.append(nn.Parameter(torch.randn(self.out_features, self.in_features * (polynomial_order + 1))))

        self.custom_norm = CustomNorm(self.out_features)
        # Layer normalization to stabilize learning and outputs
        self.layer_norms.append(nn.LayerNorm(normalized_shape=self.out_features))
        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            #nn.init.xavier_uniform_(weight, nonlinearity='linear')   # linear
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # linear
        
        for weight in self.poly_weights:
            #nn.init.xavier_uniform_(weight, nonlinearity='linear')   # linear
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # linear

        # Last normalization used before activation
        if args['final_normalization'] == 'custom_norm':
            self.final_norm = CustomNorm(self.out_features)
        elif args['final_normalization'] == 'z_score_norm':
            self.final_norm = z_score_norm
        elif args['final_normalization'] == 'z_score_norm_2d':
            self.final_norm = z_score_norm_2d

    def forward(self, x):
            # Ensure x is on the right device from the start, matching the model parameters
            x = x.to(self.base_weights[0].device)

            for i, (base_weight, poly_weight) in enumerate(zip(self.base_weights, self.poly_weights)):
                # Apply base activation to input and then linear transform with base weights
                base_output = F.linear(self.base_activation(x), base_weight)
                x_normalized = 2 * (x - x.min()) / (x.max() - x.min() +1e-8) - 1
                if self.polynomial == 'Chebyshev_1':
                    polynomial_basis = compute_chebyshev_polynomials_1(x_normalized, self.polynomial_order)
                elif self.polynomial == 'Chebyshev_2':
                    polynomial_basis = compute_chebyshev_polynomials_2(x_normalized, self.polynomial_order)
                elif self.polynomial == 'Legendre':
                    polynomial_basis = compute_legendre_polynomials(x_normalized, self.polynomial_order)
                else:
                    raise ValueError('Polynomial not implemented')
                polynomial_basis = polynomial_basis.view(x.size(0), -1)

                # Compute polynomial output using polynomial weights
                poly_output = F.linear(polynomial_basis, poly_weight)
                # Combine base and polynomial outputs, normalize, and activate
                x = self.dropout(self.base_activation(self.custom_norm(base_output + poly_output)))
            return x

class KAN_linear(nn.Module):
    def __init__(self, in_features, out_features, polynomial, lookup_table, polynomial_order=3, dropout_p=0.2):
        super(KAN_linear, self).__init__()

    # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.in_features = in_features
        self.hidden_size = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.base_activation = nn.SiLU()

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
        self.base_weights.append(nn.Parameter(torch.randn(self.hidden_size, self.in_features)))
        
        
        # Polynomial weight for handling polynomial expansions
        self.poly_weights.append(nn.Parameter(torch.randn(self.hidden_size, self.in_features * (polynomial_order + 1))))

        # Layer normalization to stabilize learning and outputs
        self.layer_norms.append(nn.LayerNorm(normalized_shape=self.hidden_size))
        self.linear = nn.Linear(self.hidden_size, self.out_features)
        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            nn.init.xavier_uniform_(weight, nonlinearity='linear')   # linear
            #nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # linear
        
        for weight in self.poly_weights:
            nn.init.xavier_uniform_(weight, nonlinearity='linear')   # linear
            #nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # linear

    def forward(self, x):
            # Ensure x is on the right device from the start, matching the model parameters
            x = x.to(self.base_weights[0].device)

            for i, (base_weight, poly_weight, layer_norm) in enumerate(zip(self.base_weights, self.poly_weights, self.layer_norms)):
                # Apply base activation to input and then linear transform with base weights
                base_output = F.linear(self.base_activation(x), base_weight)
                x_normalized = 2 * (x - x.min()) / (x.max() - x.min() +1e-8) - 1
                if self.polynomial == 'Chebyshev_1':
                    polynomial_basis = compute_chebyshev_polynomials_1(x_normalized, self.polynomial_order)
                elif self.polynomial == 'Chebyshev_2':
                    polynomial_basis = compute_chebyshev_polynomials_2(x_normalized, self.polynomial_order)
                elif self.polynomial == 'Legendre':
                    polynomial_basis = compute_legendre_polynomials(x_normalized, self.polynomial_order)
                else:
                    raise ValueError('Polynomial not implemented')
                polynomial_basis = polynomial_basis.view(x.size(0), -1)

                # Compute polynomial output using polynomial weights
                poly_output = F.linear(polynomial_basis, poly_weight)
                # Combine base and polynomial outputs, normalize, and activate
                x = self.dropout(self.base_activation(layer_norm(base_output + poly_output)))
            
            x = self.linear(x)
            return x


class KAN_linear_no_layer(nn.Module):
    def __init__(self, in_features, out_features, polynomial, polynomial_order=3, dropout_p=0.2):
        super(KAN_linear_no_layer, self).__init__()

    # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.in_features = in_features
        self.hidden_size = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.base_activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout_p)
        self.polynomial = polynomial

        # ParameterList for the polynomial weights for polynomial expansion
        self.poly_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()
        
        # Polynomial weight for handling polynomial expansions
        self.poly_weights.append(nn.Parameter(torch.randn(self.hidden_size, self.in_features * (polynomial_order + 1))))

        # Layer normalization to stabilize learning and outputs
        self.layer_norms.append(nn.LayerNorm(normalized_shape=self.hidden_size))
        self.linear = nn.Linear(self.hidden_size, self.out_features)
        
        for weight in self.poly_weights:
            nn.init.xavier_uniform_(weight, nonlinearity='linear')   # linear
            #nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # linear

    def forward(self, x):
            # Ensure x is on the right device from the start, matching the model parameters
            x = x.to(self.poly_weights[0].device)
            for i, (poly_weight, layer_norm) in enumerate(zip(self.poly_weights, self.layer_norms)):
                x_normalized = 2 * (x - x.min()) / (x.max() - x.min() +1e-8) - 1
                if self.polynomial == 'Chebyshev_1':
                    polynomial_basis = compute_chebyshev_polynomials_1(x_normalized, self.polynomial_order)
                elif self.polynomial == 'Chebyshev_2':
                    polynomial_basis = compute_chebyshev_polynomials_2(x_normalized, self.polynomial_order)
                elif self.polynomial == 'Legendre':
                    polynomial_basis = compute_legendre_polynomials(x_normalized, self.polynomial_order)
                else:
                    raise ValueError('Polynomial not implemented')
                polynomial_basis = polynomial_basis.view(x.size(0), -1)

                # Compute polynomial output using polynomial weights
                poly_output = F.linear(polynomial_basis, poly_weight)
                # Combine base and polynomial outputs, normalize, and activate
                x = self.dropout(self.base_activation(layer_norm(poly_output)))
            x = self.linear(x)
            return x



class GGB(nn.Module):
    def __init__(self, args, in_features, out_features, polynomial_order=3, dropout_p=0.2):
        super(GGB, self).__init__()  # Initialize the parent nn.Module class
        
        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.in_features = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.args = args
        
        self.base_activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_p)
        
        # ParameterList for the polynomial weights for polynomial expansion
        self.poly_weights = nn.ParameterList()
        # alpha parameter for Gegenbauer polynomials
        self.alpha_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        self.bias = nn.ParameterList()

        # Polynomial weight for handling polynomial expansions
        self.poly_weights.append(nn.Parameter(torch.randn(self.out_features, self.in_features * (polynomial_order + 1))))

        # Alpha weight for Gegenbauer polynomials
        self.alpha_weights.append(nn.Parameter(torch.randn(self.in_features, self.in_features)))

        # bias initialization
        self.bias.append(nn.Parameter(torch.randn(self.in_features)))

        # Initial activation used for alpha parameters
        self.initial_activation = nn.SiLU()
        
        # Last normalization used before activation
        if args['final_normalization'] == 'custom_norm':
            self.final_norm = CustomNorm(self.out_features)
        elif args['final_normalization'] == 'z_score_norm':
            self.final_norm = z_score_norm
        elif args['final_normalization'] == 'z_score_norm_2d':
            self.final_norm = z_score_norm_2d

        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

        for weight in self.alpha_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

        for weight in self.bias:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def forward(self, x):
        x = x.to(self.poly_weights[0].device)
        
        for i, (poly_weight, alpha_weight, bias) in enumerate(zip(self.poly_weights, self.alpha_weights, self.bias)):
            
            # Compute Gegenbauer polynomials
            x_normalized = 2 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1
            
            # Compute Gegenbauer polynomial basis
            if self.args['bias']:
                polynomial_basis = compute_gegenbauer_polynomials(x_normalized, order=self.polynomial_order, 
                                                                  alpha=alpha_weight, activation=self.initial_activation, bias=bias)
            else:
                polynomial_basis = compute_gegenbauer_polynomials(x_normalized, order=self.polynomial_order, 
                                                                  alpha=alpha_weight, activation=self.initial_activation, bias=None)
            
            polynomial_basis = polynomial_basis.view(x.size(0), -1)

            # Compute polynomial output using polynomial weights
            poly_output = F.linear(polynomial_basis, poly_weight)
            
            # Apply final activation, dropout, and normalization
            x = self.dropout(self.base_activation(self.final_norm(poly_output)))
        
        return x


class GGB_layer(nn.Module):  # Kolmogorov Arnold Network Layer with different polynomials according to lookup_table (KAN)
    def __init__(self, args, in_features, out_features, polynomial_order=3, dropout_p=0.2):
        super(GGB_layer, self).__init__()  # Initialize the parent nn.Module class
        
        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.in_features = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.args=args
        
        self.base_activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_p)
        
        # ParameterList for the base weights of each layer
        self.base_weights = nn.ParameterList()
        # ParameterList for the polynomial weights for polynomial expansion
        self.poly_weights = nn.ParameterList()
        # alpha parameter for Gegenbauer polynomials
        self.alpha_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        self.bias = nn.ParameterList()

        # Base weight for linear transformation in each layer
        self.base_weights.append(nn.Parameter(torch.randn(self.out_features, self.in_features)))
        
        # Polynomial weight for handling polynomial expansions
        self.poly_weights.append(nn.Parameter(torch.randn(self.out_features, self.in_features * (polynomial_order + 1))))

        # Alpha weight for Gegenbauer polynomials
        self.alpha_weights.append(nn.Parameter(torch.randn(self.in_features, self.in_features)))

        # bias initialization
        self.bias.append(nn.Parameter(torch.randn(self.in_features)))

        # Initial activation used for alpha parameters
        self.initial_activation = nn.SiLU()

        # Last normalization used before activation
        if args['final_normalization'] == 'custom_norm':
            self.final_norm = CustomNorm(self.out_features)
        elif args['final_normalization'] == 'z_score_norm':
            self.final_norm = z_score_norm
        elif args['final_normalization'] == 'z_score_norm_2d':
            self.final_norm = z_score_norm_2d


        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            #nn.init.xavier_uniform_(weight, nonlinearity='linear')   
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # ORIGINAL
        
        for weight in self.poly_weights:
            #nn.init.xavier_uniform_(weight, nonlinearity='linear')   
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # ORIGINAL

        for weight in self.alpha_weights:
            nn.init.xavier_uniform_(weight, nonlinearity='linear')    # ORIGINAL
            #nn.init.kaiming_uniform_(weight, nonlinearity='linear')    

        for weight in self.bias:
            nn.init.xavier_uniform_(weight, nonlinearity='linear')    # ORIGINAL
            #nn.init.kaiming_uniform_(weight, nonlinearity='linear')
    
    def forward(self, x):
        x = x.to(self.base_weights[0].device)
        
        for i, (base_weight, poly_weight, alpha_weight, bias) in enumerate(zip(self.base_weights, self.poly_weights, self.alpha_weights, self.bias)):
            
            # Linear transformation
            base_output = F.linear(self.base_activation(x), base_weight)
            
            # Compute Gegenbauer polynomials
            x_normalized = 2 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1
            
            #alpha_weight_clamped = torch.clamp(alpha_weight, min=-1.5)
            if self.args['bias']:
                polynomial_basis = compute_gegenbauer_polynomials(x_normalized, order=self.polynomial_order, 
                                                                alpha=alpha_weight, activation=self.initial_activation, bias=bias)
            else:
                polynomial_basis = compute_gegenbauer_polynomials(x_normalized, order=self.polynomial_order, 
                                                                alpha=alpha_weight, activation=self.initial_activation, bias=None)
            polynomial_basis = polynomial_basis.view(x.size(0), -1)

            # Compute polynomial output using polynomial weights
            poly_output = F.linear(polynomial_basis, poly_weight)
            
            x = self.dropout(self.base_activation(self.final_norm(base_output + poly_output)))
        
        return x

class GGB_linear(nn.Module):
    def __init__(self, args, in_features, out_features, polynomial_order=3, dropout_p=0.2):
        super(GGB_linear, self).__init__()  # Initialize the parent nn.Module class
        
        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.in_features = in_features
        self.hidden_size = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.args=args
        self.base_activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout_p)
        
        # ParameterList for the base weights of each layer
        self.base_weights = nn.ParameterList()
        # ParameterList for the polynomial weights for polynomial expansion
        self.poly_weights = nn.ParameterList()
        # alpha parameter for Gegenbauer polynomials
        self.alpha_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        # alpha bias for Gegenbauer polynomials
        self.bias = nn.ParameterList()

        # Base weight for linear transformation in each layer
        self.base_weights.append(nn.Parameter(torch.randn(self.hidden_size, self.in_features)))
        
        # Polynomial weight for handling polynomial expansions
        self.poly_weights.append(nn.Parameter(torch.randn(self.hidden_size, self.in_features * (polynomial_order + 1))))

        # Alpha weight for Gegenbauer polynomials
        self.alpha_weights.append(nn.Parameter(torch.randn(self.in_features, self.in_features)))

        # Layer normalization initialization
        self.layer_norms.append(nn.LayerNorm(normalized_shape=self.hidden_size))

        # bias initialization
        self.bias.append(nn.Parameter(torch.randn(self.in_features)))

        # Last lineal transformation
        self.linear = nn.Linear(self.hidden_size, self.out_features)

        # Initial activation used for alpha parameters
        self.initial_activation = nn.SiLU()

        # Last normalization used before activation
        if args['final_normalization'] == 'custom_norm':
            self.final_norm = CustomNorm(self.out_features)
        elif args['final_normalization'] == 'z_score_norm':
            self.final_norm = z_score_norm
        elif args['final_normalization'] == 'z_score_norm_2d':
            self.final_norm = z_score_norm_2d


        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.base_weights:
            #nn.init.xavier_uniform_(weight, nonlinearity='linear')   # linear
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # linear
        
        for weight in self.poly_weights:
            #nn.init.xavier_uniform_(weight, nonlinearity='linear')   # linear
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # linear

        for weight in self.alpha_weights:
            nn.init.xavier_uniform_(weight, nonlinearity='linear')

        for weight in self.bias:
            nn.init.xavier_uniform_(weight, nonlinearity='linear')

    def forward(self, x):
        x = x.to(self.base_weights[0].device)
        
        for i, (base_weight, poly_weight, alpha_weight, b, layer_norm) in enumerate(zip(self.base_weights, self.poly_weights, self.alpha_weights, self.bias, self.layer_norms)):
            
            # Linear transformation
            base_output = F.linear(self.base_activation(x), base_weight)
            
            # Compute Gegenbauer polynomials
            x_normalized = 2 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1
            
            #alpha_weight_clamped = torch.clamp(alpha_weight, min=-1.5)
            if self.args['bias']:
                polynomial_basis = compute_gegenbauer_polynomials(x_normalized, order=self.polynomial_order, 
                                                                alpha=alpha_weight, activation=self.initial_activation, bias=b)
            else:
                polynomial_basis = compute_gegenbauer_polynomials(x_normalized, order=self.polynomial_order, 
                                                                alpha=alpha_weight, activation=self.initial_activation, bias=None)
            polynomial_basis = polynomial_basis.view(x.size(0), -1)

            # Compute polynomial output using polynomial weights
            poly_output = F.linear(polynomial_basis, poly_weight)
            
            x = self.dropout(self.base_activation(layer_norm(base_output + poly_output)))
            x = self.linear(x)
        return x


class GGB_linear_no_layer(nn.Module):
    def __init__(self, args, in_features, out_features, polynomial_order=3, dropout_p=0.2):
        super(GGB_linear_no_layer, self).__init__()  # Initialize the parent nn.Module class
        
        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.in_features = in_features
        self.hidden_size = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which polynomials are calculated
        self.polynomial_order = polynomial_order
        # base_activation: Activation function used after each layer's computation
        self.args=args
        
        self.base_activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout_p)
        
        # ParameterList for the polynomial weights for polynomial expansion
        self.poly_weights = nn.ParameterList()
        # alpha parameter for Gegenbauer polynomials
        self.alpha_weights = nn.ParameterList()
        # ModuleList for layer normalization for each layer's output
        self.layer_norms = nn.ModuleList()

        # alpha bias for Gegenbauer polynomials
        self.bias = nn.ParameterList()
        
        # Polynomial weight for handling polynomial expansions
        self.poly_weights.append(nn.Parameter(torch.randn(self.hidden_size, self.in_features * (polynomial_order + 1))))

        # Alpha weight for Gegenbauer polynomials
        self.alpha_weights.append(nn.Parameter(torch.randn(self.in_features, self.in_features)))

        # Layer normalization initialization
        self.layer_norms.append(nn.LayerNorm(normalized_shape=self.hidden_size))

        # bias initialization
        self.bias.append(nn.Parameter(torch.randn(self.in_features)))

        # Last lineal transformation
        self.linear = nn.Linear(self.hidden_size, self.out_features)

        # Initial activation used for alpha parameters
        self.initial_activation = nn.SiLU()

        # Last normalization used before activation
        if args['final_normalization'] == 'custom_norm':
            self.final_norm = CustomNorm(self.out_features)
        elif args['final_normalization'] == 'z_score_norm':
            self.final_norm = z_score_norm
        elif args['final_normalization'] == 'z_score_norm_2d':
            self.final_norm = z_score_norm_2d


        # Initialize weights using Kaiming uniform distribution for better training start
        for weight in self.poly_weights:
            #nn.init.xavier_uniform_(weight, nonlinearity='linear')   # linear
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')   # linear

        for weight in self.alpha_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

        for weight in self.bias:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def forward(self, x):
        x = x.to(self.poly_weights[0].device)
        for i, (poly_weight, alpha_weight, b, layer_norm) in enumerate(zip(self.poly_weights, self.alpha_weights, self.bias, self.layer_norms)):
            # Compute Gegenbauer polynomials
            x_normalized = 2 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1
            
            #alpha_weight_clamped = torch.clamp(alpha_weight, min=-1.5)
            if self.args['bias']:
                polynomial_basis = compute_gegenbauer_polynomials(x_normalized, order=self.polynomial_order, 
                                                                alpha=alpha_weight, activation=self.initial_activation, bias=b)
            else:
                polynomial_basis = compute_gegenbauer_polynomials(x_normalized, order=self.polynomial_order, 
                                                                alpha=alpha_weight, activation=self.initial_activation, bias=None)
            polynomial_basis = polynomial_basis.view(x.size(0), -1)

            # Compute polynomial output using polynomial weights
            poly_output = F.linear(polynomial_basis, poly_weight)
            
            x = self.dropout(self.base_activation(layer_norm(poly_output)))
            x = self.linear(x)
        return x

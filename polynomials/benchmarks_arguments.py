import argparse
from dataset_tokenizers import *

# Initialize the parser

BENCHMARK_CLASSES = {
    'glue': GLUE_Dataset_Tokenizers,
    'super_glue': SUPERGLUE_Dataset_Tokenizers,
    'squad': SQUAD_Dataset_Tokenizers,
    'squadv2': SQUADV2_Dataset_Tokenizers,
    'swag': SWAG_Dataset_tokenizer,
    'race': RACE_Dataset_tokenizer,
    'difficult': DIFFICULT_Dataset_tokenizer
}

# dataset classes
#BENCHMARK_CLASSES[args['benchmark']](tokenizer)

def arguments(benchmark, task):
    parser = argparse.ArgumentParser(description="Deep Learning Model Parameters")

    # Data parameters
    parser.add_argument('--benchmark', type=str, default=benchmark, help='Benchmark') # [glue, superglue, squad, squad_v2, swag, race]
    parser.add_argument('--task', type=str, default=task, help='Task') # [cola, mnli, mnli_mismatched, mnli_matched, mrpc, qnli, qqp, rte, sst2, stsb, wnli, ax]
    
    
    # LM parameters
    parser.add_argument('--model_name', type=str, default="google-bert/bert-base-uncased", help='Model name') # [pysentimiento/robertuito-base-uncased, bertin-project/bertin-roberta-base-spanish, PlanTL-GOB-ES/roberta-base-bne, scjnugacj/jurisbert, xlm-roberta-base]
    parser.add_argument('--model_type', type=str, default='bert', help='Model type')

    # scheduler & optimizer parameters
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='warmup_linear', help='Learning rate scheduler')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')   #1e-5 mejor de momento, en lora en teoría el learning rate tiene que ser más alto (un orden de magnitud más alto), lo mismo para Galore
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--warmup_proportion', type=float, default=0.06, help='Warmup proportion')

    # device parameters
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--devices', type=str, default='cuda:0', help='Devices to use')


    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # KAN
    parser.add_argument('--KAN', type=bool, default=True, help='Kolgomorov Arnold for classification Head instead of MLP')
    parser.add_argument('--polynomial', type=str, default='legendre', help='Polynomial')
    parser.add_argument('--polynomial_order', type=int, default=5, help='Polynomial order')
    parser.add_argument('--lookup_table_steps', type=list, default=200000, help='Lookup table')
    parser.add_argument('--base_activation', type=str, default='nn.SiLU', help='Base activation function')
    parser.add_argument('--kan_hidden_size', type=float, default=32, help='KAN hidden size')
    parser.add_argument('--kan_dropout', type=float, default=0, help='KAN dropout')
    parser.add_argument('--initial_activation', type=str, default='silu', help='activation for alpha')   # [CustomNorm, z_score_norm, z_score_norm_2d]
    parser.add_argument('--final_normalization', type=str, default='z_score_norm', help='Custom Norm')
    parser.add_argument('--bias', type=bool, default=False, help='Bias for the alpha parameters')
    
    #Experiment
    parser.add_argument('--experiment', type=str, default='KAN_linear_no_layer', help='Experiment name')        # [KAN, KAN_layer, KAN_linear, KAN_linear_no_layer]

    # KAN: use only the KAN implementation without linear transformation in parallel neither after KAN
    # KAN_layer: use the KAN implementation with linear transformation in parallel
    # KAN_linear: use the KAN implementation with linear transformation in parallel and after KAN
    # KAN_linear_no_layer: use the KAN implementation with linear transformation after KAN but not in parallel

    #parser.add_argument('--implementation', type=str, default='Fair_baseline', help='Implementation') # [Gegenbauer, Chebyshev, Legendre, Baseline, Fair_baseline]

    # Parse the arguments
    args = parser.parse_args()
    all_data = vars(args)
    return all_data
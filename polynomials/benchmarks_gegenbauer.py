import json
from transformers import AdamW, TrainingArguments
from datasets import load_dataset

# Local imports
from Code.polynomials.datasets_utils import *
from Code.polynomials.architecture import *
from Code.polynomials.dataloader import *
from Code.polynomials.utils_gegenbauer import *
from Code.polynomials.benchmarks_arguments import *
from Code.polynomials.kan import *
from Code.polynomials.callbacks import *

BENCHMARKS = {
    'glue': ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'],
    'super_glue': ['boolq', 'cb', 'copa', 'rte', 'wic', 'wsc'], 
}

BATCH_SIZE = {
    'boolq': 6,
    'cb': 6,
    'copa': 6,
    'rte': 6,
    'wic': 6,
    'wsc': 6,
    'cola': 6,
    'mnli': 32,
    'mrpc': 6,
    'qnli': 16,
    'qqp': 32,
    'sst2': 16,
    'stsb': 6,
    'wnli': 6,
}

for benchmark in BENCHMARKS:
    for task in BENCHMARKS[benchmark]:
        args = arguments(benchmark=benchmark, task=task)
        print('Processing:', args['benchmark'], args['task'])

        model_tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
        benchmark_tokenizer = BENCHMARK_CLASSES[args['benchmark']](args['task'], model_tokenizer)

        # DATASET
        benchmark = args['benchmark']
        splits = BENCHMARK_MAPPER[benchmark][args['task']]


        if args['task'] == 'swag':
            train_dataset = load_dataset(benchmark, 'regular', split='train')
            dev_dataset = load_dataset(benchmark, 'regular', split='validation')
            train_dataset = train_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)
            dev_dataset = dev_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)
            train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']})
            dev_dataset = dev_dataset.map(lambda examples: {'labels': examples['label']})
            train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        elif args['task'] == 'race':
            train_dataset = load_dataset(benchmark, 'all',  split='train')
            dev_dataset = load_dataset(benchmark, 'all', split='validation')
            train_dataset = train_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True)
            #train_dataset = train_dataset.map(benchmark_tokenizer.relabel, batched=True)
            dev_dataset = dev_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True)
            #dev_dataset = dev_dataset.map(benchmark_tokenizer.relabel, batched=True)
            train_dataset = train_dataset.add_column('labels', train_dataset['answer'])
            dev_dataset = dev_dataset.add_column('labels', dev_dataset['answer'])
            train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        elif len(splits) == 3:

            if args['task'] == 'mnli':
                train_dataset = load_dataset(benchmark, args['task'], split='train', trust_remote_code=True)
                dev_dataset = load_dataset(benchmark, args['task'], split='validation_matched', trust_remote_code=True)

                train_dataset = train_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)
                dev_dataset = dev_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)

                train_dataset = train_dataset.add_column('labels', train_dataset['label'])
                dev_dataset = dev_dataset.add_column('labels', dev_dataset['label'])

                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            else:
                train_dataset = load_dataset(benchmark, args['task'], split='train', trust_remote_code=True)
                dev_dataset = load_dataset(benchmark, args['task'], split='validation', trust_remote_code=True)
                train_dataset = train_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)
                dev_dataset = dev_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)
                train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']})
                #train_dataset = train_dataset.add_column('labels', train_dataset['label'])
                dev_dataset = dev_dataset.map(lambda examples: {'labels': examples['label']})
                #dev_dataset = dev_dataset.add_column('labels', dev_dataset['label'])
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
                dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids','labels'])

                train_dataset = train_dataset.map(squeeze_inputs, batched=True)
                dev_dataset = dev_dataset.map(squeeze_inputs, batched=True)
            
        #print("Train dataset length:", len(train_dataset))
        # MODEL
        config = AutoConfig.from_pretrained(args['model_name'])
        num_labels = NUMBER_OF_LABELS[args['task']]
        config.num_labels = NUMBER_OF_LABELS[args['task']]
        if args['KAN']:
            config.num_labels = NUMBER_OF_LABELS[args['task']] 
            if args['implementation'] == 'Gegenbauer':
                print("Testing Gegenbauer polynomial KAN implementation")
                model = GGB_BertForSequenceClassification.from_pretrained(args['model_name'], config=config, 
                                                                    polynomial_order=args['polynomial_order'], 
                                                                    kan_dropout=args['kan_dropout'], args=args)
            elif args['implementation'] == 'Chebyshev_1':
                print("Testing Chebyshev first kind polynomial KAN implementation")
                model = KAN_BertForSequenceClassification.from_pretrained(args['model_name'], config=config,
                                                                    polynomial=args['implementation'],
                                                                    polynomial_order=args['polynomial_order'],  
                                                                    kan_dropout=args['kan_dropout'], args=args)
            
            elif args['implementation'] == 'Chebyshev_2':
                print("Testing Chebyshev second kind polynomial KAN implementation")
                model = KAN_BertForSequenceClassification.from_pretrained(args['model_name'], config=config,
                                                                    polynomial=args['implementation'],
                                                                    polynomial_order=args['polynomial_order'],  
                                                                    kan_dropout=args['kan_dropout'], args=args)
                
            elif args['implementation'] == 'Legendre':
                print("Testing Legendre KAN implementation")
                model = KAN_BertForSequenceClassification.from_pretrained(args['model_name'], config=config,
                                                                    polynomial=args['implementation'],
                                                                    polynomial_order=args['polynomial_order'],  
                                                                    kan_dropout=args['kan_dropout'], args=args)
            else:
                raise ValueError("Invalid KAN implementation")

        else:
            num_labels = NUMBER_OF_LABELS[args['task']]
            if args['implementation'] == 'Fair_baseline':
                print("Testing Fairness Baseline")
                model = FairBertForSequenceClassification.from_pretrained(args['model_name'], num_labels=config.num_labels)

            else:
                assert args['implementation'] == 'Baseline' 
                print("Testing Baseline")
                model = BertForSequenceClassification.from_pretrained(args['model_name'], num_labels=config.num_labels)


        #alpha_params = [param for name, param in model.named_parameters() if 'alpha' in name]
        classifier_params = [param for name, param in model.named_parameters() if 'classifier' in name]

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': 2e-5},
            {'params': classifier_params, 'lr': 2e-5}
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

        seed=42
        
        training_args = TrainingArguments(
            output_dir=f"/usrvol/results_hf/{args['implementation']}_BERT_FINAL_{args['experiment']}_seed_{seed}/{args['benchmark']}/{args['task']}",
            logging_dir='/usrvol/logs',
            seed=seed,
            num_train_epochs=10,
            per_device_train_batch_size= BATCH_SIZE[args['task']],
            per_device_eval_batch_size=6,
            learning_rate=2e-5,
            weight_decay=0.1,
            warmup_ratio=0.06,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-6,
            logging_steps=10,
            evaluation_strategy= "epoch",
            save_strategy= "epoch",
        )

        trainer = RollingMetricsTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            callbacks=[SaveEvaluationResultsCallback(args['benchmark'], args['task'], args['experiment'], args['implementation'], seed=seed), 
                    SaveLossesCallback(f"/usrvol/results_hf/{args['implementation']}_BERT_FINAL_{args['experiment']}_seed_{seed}/{args['benchmark']}/{args['task']}"), 
                    TimeCallback(f"/usrvol/results_hf/{args['implementation']}_BERT_FINAL_{args['experiment']}_seed_{seed}/{args['benchmark']}/{args['task']}"), 
                    GradientStatsCallback(store_stats=True, output_dir=f"/usrvol/results_hf/{args['implementation']}_BERT_FINAL_{args['experiment']}_seed_{seed}/{args['benchmark']}/{args['task']}"),
                    GeneralCalibrationCallback(output_dir=f"/usrvol/results_hf/{args['implementation']}_BERT_FINAL_{args['experiment']}_seed_{seed}/{args['benchmark']}/{args['task']}")],
            compute_metrics=compute_metrics_regression if args['task'] == 'stsb' else compute_metrics_classification,
            optimizers=(optimizer, None)
        )

        trainer.train()
        trainer.evaluate()

print("done!")
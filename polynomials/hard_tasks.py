import json
from transformers import AdamW, TrainingArguments
from datasets import load_dataset, load_from_disk

# Local imports
from datasets_utils import *
from architecture import *
from dataloader import *
from utils_gegenbauer import *
from benchmarks_arguments import *
from kan import *
from callbacks import *

BENCHMARKS = {
    'difficult': ['jigsaw-toxic-comment-classification-challenge',
                  #'yahoo_answers_topics',
                  'GoEmotions'
                  ]
}

def one_hot_labels(example):
    # Create a zero vector using a list (not PyTorch tensors)
    one_hot_vector = [0.0] * config.num_labels  # Python list with float values
    if args['task'] == 'jigsaw-toxic-comment-classification-challenge':
        specific_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        for i, label in enumerate(specific_labels):
            if example[label] == 1:
                one_hot_vector[i] = 1.0
    elif args['task'] == 'GoEmotions':
        for idx in example['labels']:  # Assume 'labels' is a list of indices
            one_hot_vector[idx] = 1.0
    example['labels'] = one_hot_vector  # Set the labels to the one-hot vector
    return example

BATCH_SIZE = {
    'jigsaw-toxic-comment-classification-challenge': 32,
    'yahoo_answers_topics':48,
    'GoEmotions':32
}

for benchmark in BENCHMARKS:
    for task in BENCHMARKS[benchmark]:
        args = arguments(benchmark=benchmark, task=task)
        print('Processing:', args['benchmark'], args['task'])

        model_tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
        benchmark_tokenizer = BENCHMARK_CLASSES[args['benchmark']](args['task'], model_tokenizer)

        # LABELS
        config = AutoConfig.from_pretrained(args['model_name'])
        num_labels = NUMBER_OF_LABELS[args['task']]
        config.num_labels = NUMBER_OF_LABELS[args['task']]
        
        # DATASET
        benchmark = args['benchmark']
        splits = BENCHMARK_MAPPER[benchmark][args['task']]

        if args['task'] == 'GoEmotions':
            config.problem_type = 'multi_label_classification'
            train_dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split='train', trust_remote_code=True)
            dev_dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split='validation', trust_remote_code=True)
            test_dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split='test', trust_remote_code=True)

            train_dataset = train_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)
            dev_dataset = dev_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)
            test_dataset = test_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True, load_from_cache_file=True)

            train_dataset = train_dataset.map(cast_labels_to_float, load_from_cache_file=False)
            dev_dataset = dev_dataset.map(cast_labels_to_float, load_from_cache_file=False)
            test_dataset = test_dataset.map(cast_labels_to_float, load_from_cache_file=False)


            # train_dataset = train_dataset.add_column('labels', train_dataset['label'])
            # dev_dataset = dev_dataset.add_column('labels', dev_dataset['label'])
            # test_dataset = test_dataset.add_column('labels', test_dataset['label'])

            train_dataset = train_dataset.map(one_hot_labels, load_from_cache_file=False)
            dev_dataset = dev_dataset.map(one_hot_labels, load_from_cache_file=False)
            test_dataset = test_dataset.map(one_hot_labels, load_from_cache_file=False)

            train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        elif args['task'] == 'yahoo_answers_topics':
            train_dataset = load_from_disk(dataset_path=f"/usrvol/data/{task}_reduced/train")
            test_dataset = load_from_disk(dataset_path=f"/usrvol/data/{task}_reduced/test")
            
            train_dataset = train_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True)
            test_dataset = test_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True)
            
            train_dataset = train_dataset.add_column('labels', train_dataset['topic'])
            test_dataset = test_dataset.add_column('labels', test_dataset['topic'])
            
            # Set the dataset format
            train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            
        elif args['task'] == 'jigsaw-toxic-comment-classification-challenge':
            config.problem_type = 'multi_label_classification'
            train_dataset = load_from_disk(dataset_path=f"/usrvol/data/{task}_reduced/train")
            test_dataset = load_from_disk(dataset_path=f"/usrvol/data/{task}_reduced/test")
            
            train_dataset = train_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True)
            test_dataset = test_dataset.map(benchmark_tokenizer.get_tokenizer(), batched=True)
            
            train_dataset = train_dataset.map(one_hot_labels)
            test_dataset = test_dataset.map(one_hot_labels)
            
            # Set the dataset format
            train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        #print("Train dataset length:", len(train_dataset))
        # MODEL
        else:
            raise ValueError("Invalid task")
        
        if args['KAN']: 
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
            if args['implementation'] == 'Fair_baseline':
                print("Testing Fair Baseline")
                model = FairBertForSequenceClassification.from_pretrained(args['model_name'],config=config)

            else:
                assert args['implementation'] == 'Baseline' 
                print("Testing Baseline")
                model = BertForSequenceClassification.from_pretrained(args['model_name'], config=config)


            #alpha_params = [param for name, param in model.named_parameters() if 'alpha' in name]
            classifier_params = [param for name, param in model.named_parameters() if 'classifier' in name]

            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': 2e-5},
                {'params': classifier_params, 'lr': 2e-5}
            ]

            optimizer = AdamW(optimizer_grouped_parameters)

            training_args = TrainingArguments(
                output_dir=f"/usrvol/results_hf/{args['implementation']}_BERT_FINAL_{args['experiment']}/{args['benchmark']}/{args['task']}",
                logging_dir='/usrvol/logs',
                seed=42,
                num_train_epochs=10,
                per_device_train_batch_size= BATCH_SIZE[args['task']],
                per_device_eval_batch_size=4,
                learning_rate=2e-5,
                weight_decay=0.1,
                warmup_ratio=0.06,
                adam_beta1=0.9,
                adam_beta2=0.98,
                adam_epsilon=1e-6,
                logging_steps=10,
                evaluation_strategy= "epoch",
                save_strategy= "epoch",
                logging_strategy= "steps",
            )

            trainer = RollingMetricsTrainer(
                model=model,
                args=training_args,
                train_dataset= train_dataset,
                eval_dataset= test_dataset,
                callbacks=[SaveEvaluationResultsCallback(args['benchmark'], args['task'], args['experiment'], args['implementation']), 
                        SaveLossesCallback(f"/usrvol/results_hf/{args['implementation']}_BERT_FINAL_{args['experiment']}/{args['benchmark']}/{args['task']}"), 
                        TimeCallback(f"/usrvol/results_hf/{args['implementation']}_BERT_FINAL_{args['experiment']}/{args['benchmark']}/{args['task']}"), 
                        GeneralCalibrationCallback(output_dir=f"/usrvol/results_hf/{args['implementation']}_BERT_FINAL_{args['experiment']}/{args['benchmark']}/{args['task']}")],
                compute_metrics=compute_metrics_regression if args['task'] == 'stsb' else compute_metrics_classification,
                optimizers=(optimizer, None)
            )

            trainer.train()
            #trainer.evaluate()

print("done!")
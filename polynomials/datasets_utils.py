import torch

GLUE_TASKS_SPLITS = {
    'cola': ['train', 'validation', 'test'],
    'mnli': ['train', 'validation_matched', 'validation_mismatched'],
    'mrpc': ['train', 'validation', 'test'],
    'qnli': ['train', 'validation', 'test'],
    'qqp': ['train', 'validation', 'test'],
    'rte': ['train', 'validation', 'test'],
    'sst2': ['train', 'validation', 'test'],
    'stsb': ['train', 'validation', 'test'],
    'wnli': ['train', 'validation', 'test'],
}

SUPERGLUE_TASKS_SPLITS = {
    'boolq': ['train', 'validation', 'test'],
    'cb': ['train', 'validation', 'test'],
    'copa': ['train', 'validation', 'test'],
    'rte': ['train', 'validation', 'test'],
    'wic': ['train', 'validation', 'test'],
    'wsc': ['train', 'validation', 'test'],
}


SQUAD_TASKS_SPLITS = { 
    'squad': ['train', 'validation'],
    'squad_v2': ['train', 'validation']
}


SWAG_TASKS_SPLITS = {
    'swag': ['train', 'validation']
}

RACE_TASKS_SPLITS = {   
    'race': ['train', 'validation', 'test']
}

DIFFICULT_TASKS_SPLITS = {
    'jigsaw-toxic-comment-classification-challenge': ['train', 'test'],
    'yahoo_answers_topics': ['train', 'test'],
    'GoEmotions': ['train', 'validation', 'test']
}


BENCHMARK_MAPPER = {
    'glue': GLUE_TASKS_SPLITS,
    'super_glue': SUPERGLUE_TASKS_SPLITS,
    'squad': SQUAD_TASKS_SPLITS,
    'squadv2': SQUAD_TASKS_SPLITS,
    'swag': SWAG_TASKS_SPLITS,
    'race': RACE_TASKS_SPLITS,
    'difficult': DIFFICULT_TASKS_SPLITS
}


NUMBER_OF_LABELS = {
    'cola': 2,
    'mnli': 3,
    'mrpc': 2,
    'qnli': 2,
    'qqp': 2,
    'rte': 2,
    'sst2': 2,
    'stsb': 1,
    'wnli': 2,
    'ax': 2,
    'boolq': 2,
    'cb': 3,
    'copa': 2,
    'wic': 2,
    'wsc': 2,
    'squad': 'passage',
    'squad_v2': 'passage',
    'swag': 4,
    'race': 4 ,
    'jigsaw-toxic-comment-classification-challenge': 6,
    'GoEmotions': 28,
    'yahoo_answers_topics': 10
}

# type of problem according to huggingface
PROBLEM_TYPE = {
    'cola': 'binary_classification',
    'mnli': 'multi_class_classification',
    'mrpc': 'binary_classification',
    'qnli': 'binary_classification',
    'qqp': 'binary_classification',
    'rte': 'binary_classification',
    'sst2': 'binary_classification',
    'stsb': 'regression',
    'wnli': 'binary_classification',
    'ax': 'binary_classification',
    'boolq': 'binary_classification',
    'cb': 'multi_class_classification',
    'copa': 'binary_classification',
    'wic': 'binary_classification',
    'wsc': 'binary_classification',
    'squad': 'extractive_qa',
    'squad_v2': 'extractive_qa',
    'swag': 'binary_classification',
    'race': 'binary_classification',
    'jigsaw-toxic-comment-classification-challenge': 'multi_label_classification',
    'GoEmotions': 'multi_label_classification',
    'yahoo_answers_topics': 'multi_class_classification',
}

LOSS_FUNCTIONS = {
    'binary_classification': torch.nn.BCEWithLogitsLoss(),
    'multi_class_classification': torch.nn.CrossEntropyLoss(),
    'multi_label_classification': torch.nn.BCEWithLogitsLoss(),
    'regression': torch.nn.MSELoss(),
    'extractive_qa': torch.nn.CrossEntropyLoss()
}

DATASETS_FOLDERS = {
    'glue_rte':'/usrvol/data/glue/rte',
    'glue_cola': '/usrvol/data/glue/cola',
    'glue_mnli':'/usrvol/data/glue/mnli',
    'glue_mrpc':'/usrvol/data/glue/mrpc',
    'glue_qnli':'/usrvol/data/glue/qnli',
    'glue_qqp':'/usrvol/data/glue/qqp',
    'glue_sst2':'/usrvol/data/glue/sst2',
    'glue_stsb':'/usrvol/data/glue/stsb',
    'glue_wnli':'/usrvol/data/glue/wnli',
    'super_glue_boolq':'/usrvol/data/super_glue/boolq',
    'super_glue_cb':'/usrvol/data/super_glue/cb',
    'super_glue_copa':'/usrvol/data/super_glue/copa',
    'super_glue_rte':'/usrvol/data/super_glue/rte',
    'super_glue_wic':'/usrvol/data/super_glue/wic',
    'super_glue_wsc':'/usrvol/data/super_glue/wsc',
    'squad_squad':'/usrvol/data/squad/',
    'squad_v2_squad_v2':'/usrvol/data/squadv2/',
    'swag_swag':'/usrvol/data/swag/',
    'race_race' : '/usrvol/data/race/'
}
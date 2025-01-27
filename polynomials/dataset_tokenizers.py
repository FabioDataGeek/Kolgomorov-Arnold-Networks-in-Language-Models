from transformers import AutoTokenizer


LABEL_MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}



def squeeze_inputs(example):
    example['input_ids'] = example['input_ids'].squeeze(1)
    example['attention_mask'] = example['attention_mask'].squeeze(1)
    example['token_type_ids'] = example['token_type_ids'].squeeze(1)
    return example


def convert_labels_to_numbers(examples):
    for example in range(len(examples['answer'])):
        examples['answer'][example] = LABEL_MAPPING[examples['answer'][example]]
    return examples


# FOR GLUE DATASET
class GLUE_Dataset_Tokenizers():
    def __init__(self, dataset_name, tokenizer):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        #self.tokenizer.model_max_length = 128  # Set the tokenizer's max length to 128

    def tokenize_ax(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['premise'], examples['hypothesis'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_qnli(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['question'], examples['sentence'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_rte(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        tokenized_examples = self.tokenizer(examples['sentence1'], examples['sentence2'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)
        return tokenized_examples

    # Binary Classification
    def tokenize_wnli(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['sentence1'], examples['sentence2'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_sst2(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['sentence'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_mrpc(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['sentence1'], examples['sentence2'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_cola(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['sentence'], 
                            truncation=True, return_tensors='pt',
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_qqp(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['question1'], examples['question2'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Multiclass Classification
    def tokenize_mnli(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['premise'], examples['hypothesis'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Regression
    def tokenize_stsb(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['sentence1'], examples['sentence2'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    GLUE_DATASET_DICT = {
    'qnli': tokenize_qnli,
    'rte': tokenize_rte,
    'wnli': tokenize_wnli,
    'sst2': tokenize_sst2,
    'mrpc': tokenize_mrpc,
    'cola': tokenize_cola,
    'qqp': tokenize_qqp,
    'mnli': tokenize_mnli,
    'stsb': tokenize_stsb,
    'ax': tokenize_ax
    }

    def get_tokenizer(self):
        tokenize_func = self.GLUE_DATASET_DICT[self.dataset_name]
        def wrapper(examples):
            #a = tokenize_func(self, examples)
            return tokenize_func(self, examples)
        return wrapper
    

# FOR SUPERGLUE DATASET
class SUPERGLUE_Dataset_Tokenizers():
    def __init__(self, dataset_name, tokenizer):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

    # Binary Classification
    def tokenize_boolq(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['question'], examples['passage'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Multiclass Classification
    def tokenize_cb(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['premise'], examples['hypothesis'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_copa(self, examples):

        texts1 = []
        texts2 = []
        for p, c1, c2 in zip(examples['premise'], examples['choice1'], examples['choice2']):
            if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                return_token_type_ids = True
                texts1.append(p)
                texts2.append(c1 + '[SEP]' + c2)
            else:
                return_token_type_ids = False
                texts1.append(p)
                texts2.append(c1 + ' ' + c2)
        tokenized_texts = self.tokenizer(texts1, texts2, truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

        # Squeeze the tensors
        for key in tokenized_texts.keys():
            tokenized_texts[key] = tokenized_texts[key].squeeze(1)
    
        return tokenized_texts



    # Multilabel Classification
    """ def tokenize_multirc(self, examples):
        return self.tokenizer(examples['paragraph'], examples['question'], examples['answer'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=True) """

    # This one needs further adaptation
    """ def tokenize_record(examples):
        return tokenizer(examples['passage'], examples['question'], examples['answer'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=True) """

    # Binary Classification
    def tokenize_rte(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['premise'], examples['hypothesis'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_wic(self, examples):
        texts1 = []
        texts2 = []
        for w, s1, s2, label in zip(examples['word'], examples['sentence1'], examples['sentence2'], examples['label']):
            if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                return_token_type_ids = True
                texts1.append(w)
                texts2.append(s1 + '[SEP]' + s2)
            else:
                return_token_type_ids = False
                texts1.append(w)
                texts2.append(s1 + ' ' + s2)
        
        
        return self.tokenizer(texts1, texts2, truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    # Binary Classification
    def tokenize_wsc(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['text'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)
    
    SUPERGLUE_DATASET_DICT = {
        'boolq': tokenize_boolq,
        'cb': tokenize_cb,
        'copa': tokenize_copa,
        #'multirc': tokenize_multirc,
        #'record': tokenize_record,
        'rte': tokenize_rte,
        'wic': tokenize_wic,
        'wsc': tokenize_wsc
    }

    def get_tokenizer(self):
        tokenize_func = self.SUPERGLUE_DATASET_DICT[self.dataset_name]
        def wrapper(examples):
            return tokenize_func(self, examples)
        return wrapper
    
    

# FOR SQUAD DATASET
class SQUAD_Dataset_Tokenizers():
    def __init__(self, dataset_name, tokenizer):
        self.tokenizer = tokenizer
    def tokenize_squad(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False

        return self.tokenizer(examples['question'], examples['context'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)
    
    def get_tokenizer(self):
        tokenize_func = self.tokenize_squad
        def wrapper(examples):
            return tokenize_func(self, examples)
        return wrapper


# FOR SQUADV2 DATASET
class SQUADV2_Dataset_Tokenizers():
    def __init__(self, dataset_name, tokenizer):
        self.tokenizer = tokenizer  
    
    def tokenize_squad(self, examples):
        if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
            return_token_type_ids = True
        else:
            return_token_type_ids = False
        return self.tokenizer(examples['question'], examples['context'], 
                            truncation=True, return_tensors='pt',  
                            padding='max_length', max_length=512, 
                            return_attention_mask=True, return_token_type_ids=return_token_type_ids)
    
    def get_tokenizer(self):
        tokenize_func = self.tokenize_squad
        def wrapper(examples):
            return tokenize_func(self, examples)
        return wrapper
    

# FOR SWAG DATASET
class SWAG_Dataset_tokenizer():
    def __init__(self, dataset_name, tokenizer):
        self.tokenizer = tokenizer

    def tokenizer_swag(self, examples):
        texts1 = []
        texts2 = []
        for s, e0, e1, e2, e3 in zip(examples['startphrase'], examples['ending0'], examples['ending1'], examples['ending2'], examples['ending3']):
            if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                return_token_type_ids = True
                texts1.append(s)
                texts2.append(e0 + '[SEP]' + e1 + '[SEP]' + e2 + '[SEP]' + e3)
            else:
                return_token_type_ids = False
                texts1.append(s)
                texts2.append(e0 + ' ' + e1 + ' ' + e2 + ' ' + e3)

        return self.tokenizer(texts1, texts2, truncation=True, return_tensors='pt',  
                        padding='max_length', max_length=512, 
                        return_attention_mask=True, return_token_type_ids=return_token_type_ids)
    
    def get_tokenizer(self):
        tokenize_func = self.tokenizer_swag
        def wrapper(examples):
            return tokenize_func(examples)
        return wrapper
    

# FOR RACE DATASET
class RACE_Dataset_tokenizer():
    def __init__(self, dataset_name, tokenizer):
        self.tokenizer = tokenizer

    def tokenizer_race(self, examples):
        examples = convert_labels_to_numbers(examples)
        
        texts1 = []
        texts2 = []

        for a, q, o in zip(examples['article'], examples['question'], examples['options']):
            if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                return_token_type_ids = True
                texts1.append(a + '[SEP]' + q)
                texts2.append(o[0] + '[SEP]' + o[1] + '[SEP]' + o[2] + '[SEP]' + o[3])
            else:
                return_token_type_ids = False
                texts1.append(a + ' ' + q)
                texts2.append(o[0] + ' ' + o[1] + ' ' + o[2] + ' ' + o[3])
        

        return self.tokenizer(texts1, texts2, truncation=True, return_tensors='pt',  
                        padding='max_length', max_length=512, 
                        return_attention_mask=True, return_token_type_ids=return_token_type_ids)

    def relabel(self, examples):
        for example in range(len(examples)):
            label = examples['answer'][example]
        if label in LABEL_MAPPING:
            examples['answer'][example] = LABEL_MAPPING[label]
        else:
            print(f"Unexpected label: {label}")
        return examples

    def get_tokenizer(self):
        tokenize_func = self.tokenizer_race
        def wrapper(examples):
            return tokenize_func(examples)
        return wrapper
    
class DIFFICULT_Dataset_tokenizer():
    def __init__(self, dataset_name, tokenizer):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        
    def tokenizer_jigsaw(self, examples):
        return self.tokenizer(examples['comment_text'], truncation=True, return_tensors='pt',  
                        padding='max_length', max_length=512, 
                        return_attention_mask=True, return_token_type_ids=False)
        
    def tokenizer_GoEmotions(self, examples):
        return self.tokenizer(examples['text'], truncation=True, return_tensors='pt',  
                        padding='max_length', max_length=512, 
                        return_attention_mask=True, return_token_type_ids=False)
        
    def tokenizer_yahoo_answers(self, examples):
        # Concatenate corresponding elements in a batch-wise manner
        combined_texts = [
            title + ' [SEP] ' + content + ' [SEP] ' + answer
            for title, content, answer in zip(examples['question_title'], examples['question_content'], examples['best_answer'])
        ]
        # Tokenize the combined texts
        return self.tokenizer(
            combined_texts,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512,
            return_attention_mask=True,
            return_token_type_ids=False
        )
    
    DIFFICULT_DATASET_DICT = {
        'jigsaw-toxic-comment-classification-challenge': tokenizer_jigsaw,
        'GoEmotions': tokenizer_GoEmotions,
        'yahoo_answers_topics': tokenizer_yahoo_answers
    
    }
    
    def get_tokenizer(self):
        tokenize_func = self.DIFFICULT_DATASET_DICT[self.dataset_name]
        def wrapper(examples):
            #a = tokenize_func(self, examples)
            return tokenize_func(self, examples)
        return wrapper
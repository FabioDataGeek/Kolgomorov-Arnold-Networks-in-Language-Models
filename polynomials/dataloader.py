import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import json

LABEL_TO_INT = {'directiva': 0, 'comisiva': 1, 'expresiva': 2, 'representativa': 3}
LABEL_DICT = {'contradiction': 0, 'entailment': 1, 'neutral': 2, '-': 3}

LABEL_MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}



class BenchmarksDataset(Dataset):
    def __init__(self, json_file, tokenizer, benchmark, task=None):
        
        with open(json_file, 'r') as f:
            json_file = json.load(f)
        
        self.data = json_file
        self.tokenizer = tokenizer
        self.benchmark = benchmark
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.benchmark == 'glue':

            if self.task == 'rte':
                sentence1 = self.data[idx]['sentence1']
                sentence2 = self.data[idx]['sentence2']
                inputs = self.tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),  # Remove the batch dimension
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)  # No need for an extra dimension
                }
            
            elif self.task == 'qnli':
                question = self.data[idx]['question']
                sentence = self.data[idx]['sentence']
                inputs = self.tokenizer(question, sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }
            
            elif self.task == 'wnli':
                sentence1 = self.data[idx]['sentence1']
                sentence2 = self.data[idx]['sentence2']
                inputs = self.tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }
            
            elif self.task == 'sst2':
                sentence1 = self.data[idx]['sentence']
                inputs = self.tokenizer(sentence1, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }
            
            elif self.task == 'mrpc':
                sentence1 = self.data[idx]['sentence1']
                sentence2 = self.data[idx]['sentence2']
                inputs = self.tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }
            
            elif self.task == 'cola':
                sentence = self.data[idx]['sentence']
                inputs = self.tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }
            
            elif self.task == 'qqp':
                question1 = self.data[idx]['question1']
                question2 = self.data[idx]['question2']
                inputs = self.tokenizer(question1, question2, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }
            
            elif self.task == 'mnli':
                premise = self.data[idx]['premise']
                hypothesis = self.data[idx]['hypothesis']
                inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }
            
            elif self.task == 'stsb':
                sentence1 = self.data[idx]['sentence1']
                sentence2 = self.data[idx]['sentence2']
                inputs = self.tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }

            else:
                raise ValueError('Task not recognized')
        
        elif self.benchmark == 'super_glue':
            
            if self.task == 'boolqa':
                question = self.data[idx]['question']
                passage = self.data[idx]['passage']
                inputs = self.tokenizer(question, passage, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }

            elif self.task == 'cb':
                premise = self.data[idx]['premise']
                hypothesis = self.data[idx]['hypothesis']
                inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }

            elif self.task == 'copa':
                premise = self.data[idx]['premise']
                choice1 = self.data[idx]['choice1']
                choice2 = self.data[idx]['choice2']
                if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                    return_token_type_ids = True
                    text = premise + '[SEP]' + choice1 + '[SEP]' + choice2
                else:
                    return_token_type_ids = False
                    text = premise + ' ' + choice1 + ' ' + choice2 
                
                inputs = self.tokenizer(text, truncation=True, return_tensors='pt', padding='max_length', max_length=512, return_attention_mask=True, return_token_type_ids=return_token_type_ids)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }

            elif self.task == 'rte':
                premise = self.data[idx]['premise']
                hypothesis = self.data[idx]['hypothesis']
                label = self.data[idx]['label']
                inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }
            
            elif self.task == 'wic':
                word = self.data[idx]['word']
                sentence1 = self.data[idx]['sentence1']
                sentence2 = self.data[idx]['sentence2']
                if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                    return_token_type_ids = True
                    text = word + '[SEP]' + sentence1 + '[SEP]' + sentence2
                else:
                    return_token_type_ids = False
                    text = word + ' ' + sentence1 + ' ' + sentence2

                inputs = self.tokenizer(text, truncation=True, return_tensors='pt', padding='max_length', max_length=512, return_attention_mask=True, return_token_type_ids=return_token_type_ids)
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }

            elif self.task == 'wsc':
                text = self.data[idx]['text']
                inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
                label = self.data[idx]['label']
                return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)
                }

            else:
                raise ValueError('Task not recognized')
        
        elif self.benchmark == 'squad' or self.benchmark == 'squad_v2':
            context = self.data[idx]['context']
            question = self.data[idx]['question']
            inputs = self.tokenizer(context, question, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True, return_token_type_ids=True)
            label = self.data[idx]['label']
            return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs['token_type_ids'].squeeze(0),
            'labels': torch.tensor(label)
            }
        
        elif self.benchmark == 'swag':
            startphrase = self.data[idx]['startphrase']
            ending0 = self.data[idx]['ending0']
            ending1 = self.data[idx]['ending1']
            ending2 = self.data[idx]['ending2']
            ending3 = self.data[idx]['ending3']

            if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                return_token_type_ids = True
                text = startphrase + '[SEP]' + ending0 + '[SEP]' + ending1 + '[SEP]' + ending2 + '[SEP]' + ending3 
            else:
                return_token_type_ids = False
                text = startphrase + ' ' + ending0 + ' ' + ending1 + ' ' + ending2 + ' ' + ending3
            
            inputs = self.tokenizer(text, truncation=True, return_tensors='pt', padding='max_length', max_length=512, return_attention_mask=True, return_token_type_ids=return_token_type_ids)
            label = self.data[idx]['label']
            return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs['token_type_ids'].squeeze(0),
            'labels': torch.tensor(label)
            }
        
        elif self.benchmark == 'race':
            article = self.data[idx]['article']
            question = self.data[idx]['question']
            options = self.data[idx]['options']
            option1 = options[0]
            option2 = options[1]
            option3 = options[2]
            option4 = options[3]
            answer = self.data[idx]['answer']
            
            if self.tokenizer.name_or_path == 'google-bert/bert-base-uncased':
                return_token_type_ids = True
                text = article + '[SEP]' + question + '[SEP]' + option1 + '[SEP]' + option2 + '[SEP]' + option3 + '[SEP]' + option4
            else:
                return_token_type_ids = False
                text = article + ' ' + question + ' ' + option1 + ' ' + option2 + ' ' + option3 + ' ' + option4
            label = LABEL_MAPPING[answer]
            inputs = self.tokenizer(text, truncation=True, return_tensors='pt', padding='max_length', max_length=512, 
                                    return_attention_mask=True, return_token_type_ids=return_token_type_ids)

            return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs['token_type_ids'].squeeze(0),
            'labels': torch.tensor(label)
            }

# Create Pytorch Dataset
class CustomDataset(Dataset):
    def __init__(self, labels, text_data, model_name, npn_statements=None, emo_statements=None):
        self.text_data = text_data
        self.npn_statements = npn_statements
        self.emo_statements = emo_statements
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        self.encodings = [self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=128) for text in self.text_data]
    def __len__(self):
        return len(self.labels)

    # Here we consider a different dataset for the architecture if we have npn_statements and/or emo_statements
    def __getitem__(self, idx):
        tokenized_text = self.encodings[idx]
        input_ids = tokenized_text.input_ids
        input_ids = input_ids.squeeze(0)
        attention_mask = tokenized_text.attention_mask
        attention_mask = attention_mask.squeeze(0)
        labels = torch.tensor(LABEL_TO_INT[self.labels[idx]], dtype=torch.long)
        if self.npn_statements is not None:
            if self.emo_statements is not None:
                npn_statements = torch.tensor(self.npn_statements[idx], dtype= torch.float32)
                emo_statements = torch.tensor(self.emo_statements[idx], dtype= torch.float32)
                return input_ids, attention_mask, npn_statements, emo_statements, labels
            else:
                npn_statements = torch.tensor(self.npn_statements[idx], dtype= torch.float32)
                return input_ids, attention_mask, npn_statements, labels
        elif self.emo_statements is not None:
            emo_statements = torch.tensor(self.emo_statements[idx], dtype= torch.float32)
            return input_ids, attention_mask, emo_statements, labels
        else:
            return input_ids, attention_mask , labels
        

class Dataset_LLM(Dataset):
    def __init__(self, file, tokenizer):
        with open(file, 'rb') as f:
            self.data = pkl.load(f)
            self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1 = self.data[idx][0]
        sentence2 = self.data[idx][1]
        inputs = self.tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', truncation=True, max_length=300, return_attention_mask=True, return_token_type_ids=True)
        label = LABEL_DICT[self.data[idx][2]]
        
        return {
                'input_ids': inputs['input_ids'].squeeze(0),  # Remove the batch dimension
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'token_type_ids': inputs['token_type_ids'].squeeze(0),
                'labels': torch.tensor(label)  # No need for an extra dimension
            }
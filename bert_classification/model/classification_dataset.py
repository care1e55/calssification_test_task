import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.device = torch.device('cuda', 1)
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_plus = [ 
            tokenizer.encode_plus(item, max_length=max_seq_length, pad_to_max_length=True) 
            for item in data['sentence']
        ]
        self.input_ids = torch.tensor([ i['input_ids'] for i in self.encoded_plus ], dtype=torch.long) 
        self.attention_mask = torch.tensor([ i['attention_mask'] for i in self.encoded_plus ], dtype=torch.long) 
        self.token_type_ids = torch.tensor([ i['token_type_ids'] for i in self.encoded_plus ], dtype=torch.long) 
        self.target = torch.tensor(list(self.data['label']), dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx].to(self.device),
            'attention_mask': self.attention_mask[idx].to(self.device),
            'targets': self.target[idx].to(self.device)
        }

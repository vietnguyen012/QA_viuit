from torch.utils.data import DataLoader
from data_loader import pack_sequence
import torch
class TestDataloader(DataLoader):
  def __init__(self,data_set,shuffle=False,device="cuda",batch_size=16):
        super(TestDataloader,self).__init__(dataset=data_set,collate_fn=self.collate_fn,shuffle=shuffle,batch_size=batch_size)
        self.device = device
  def collate_fn(self,batch):
        ids = []
        attention_mask = []
        token_type_ids = []
        input_ids=[]
        question_ids = []
        question_mask = []
        question_token_types = []
        for item in batch:
          ids.append(item.id)
          input_ids.append(item.input_ids)
          attention_mask.append(item.mask_attention)
          token_type_ids.append(item.token_types)
          question_ids.append(item.question_id)
          question_mask.append(item.question_mask)
          question_token_types.append(item.question_token_type)
        question_ids = pack_sequence(question_ids,0)
        question_mask = pack_sequence(question_mask,0)
        question_token_types = pack_sequence(question_token_types,0)
        return {
            "ids":ids,
            "input_ids":torch.LongTensor(input_ids).to(self.device),
            "attention_mask": torch.LongTensor(attention_mask).to(self.device),
            "token_type_ids":torch.LongTensor(token_type_ids).to(self.device),
            "question_ids": question_ids.to(self.device),
            "question_mask": question_mask.to(self.device),
            "question_token_types":question_token_types.to(self.device)
        }
from torch.utils.data import DataLoader
from data import QA_data,convert_to_input_feature
import torch 

def pack_sequence(ids,pad):
    max_length = max(len(x) for x in ids)
    packed_list = []
    for id in ids:
        if len(id) < max_length:
            packed_list.append(id+[pad]*(max_length-len(id)))
        else:
            packed_list.append(id)
    return torch.LongTensor(packed_list)

class QA_Dataloader(DataLoader):
    def __init__(self,data_set,shuffle=False,device="cuda",batch_size=16):
        super(QA_Dataloader,self).__init__(dataset=data_set,collate_fn=self.collate_fn,shuffle=shuffle,batch_size=batch_size)
        self.device = device
    def collate_fn(self,batch):
        ids = []
        offset_mappings = []
        attention_mask = []
        token_type_ids = []
        start_position = []
        end_position = []
        has_answer = []
        # masked_pos = []
        # masked_tokens = []
        question_ids = []
        question_mask = []
        question_token_types = []
        input_ids=[]
        sentence_ans_ids = []
        masked_sentence_ans_ids = []
        for item in batch:
            ids.append(item.id)
            input_ids.append(item.input_ids)
            offset_mappings.append(item.offset_mapping)
            attention_mask.append(item.attention_mask)
            token_type_ids.append(item.token_type_ids)
            start_position.append(item.start_position)
            end_position.append(item.end_position)
            has_answer.append(item.has_answer)
            # masked_tokens.append(item.masked_tokens)
            # masked_pos.append(item.masked_pos)
            question_ids.append(item.question_id)
            question_mask.append(item.question_mask)
            question_token_types.append(item.question_token_type)
            sentence_ans_ids.append(item.sentence_ans_ids)
            masked_sentence_ans_ids.append(item.masked_sentence_ans_ids)
        sentence_ans_ids = pack_sequence(sentence_ans_ids,0)
        masked_sentence_ans_ids = pack_sequence(masked_sentence_ans_ids,0)
        question_ids = pack_sequence(question_ids,0)
        question_mask = pack_sequence(question_mask,0)
        question_token_types = pack_sequence(question_token_types,0)
        return {
            "ids":ids,
            "input_ids": torch.LongTensor(input_ids).to(self.device),
            "offset_mappings":offset_mappings,
            "attention_mask": torch.LongTensor(attention_mask).to(self.device),
            "token_type_ids": torch.LongTensor(token_type_ids).to(self.device),
            "start_position": torch.LongTensor(start_position).to(self.device),
            "end_position": torch.LongTensor(end_position).to(self.device),
            "has_answer": torch.LongTensor(has_answer).to(self.device),
            # "masked_tokens" : torch.LongTensor(masked_tokens).to(self.device),
            # "masked_pos": torch.LongTensor(masked_pos).to(self.device),
            "sentence_ans_ids": sentence_ans_ids.to(self.device),
            "masked_sentence_ans_ids": masked_sentence_ans_ids.to(self.device),
            "question_ids": question_ids.to(self.device),
            "question_mask": question_mask.to(self.device),
            "question_token_types":question_token_types.to(self.device)
        }

if __name__ == "__main__":

    examples = QA_data(128,384,"train.json",max_masked_pred=20)
    input_feature_list = []
    for ex in examples:
        input_feature_list = convert_to_input_feature(ex,input_feature_list)

    data_loader = QA_Dataloader(input_feature_list,shuffle=True,device="cuda",batch_size=3)
    for batch in data_loader:
        print(batch)
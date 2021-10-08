from torch.utils.data import Dataset 
import json
from transformers import AutoTokenizer
from random import shuffle,randint
from random import random
from collections import defaultdict

class example:
    def __init__(self,id,question,start_pos,ans,is_impossible,context):
        self.id = id
        self.context = context
        self.question = question
        self.start_pos = start_pos
        self.ans = ans 
        self.is_impossible = is_impossible

class QA_data(Dataset):
    def __init__(self,doc_stride,max_length,file_name,tokenizer_name="FPTAI/velectra-base-discriminator-cased",max_masked_pred=30):
        super().__init__()
        self.examples = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,do_lower_case=False)
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_masked_pred = max_masked_pred
        self.read_file(file_name)

    def read_file(self,file_json):
        len_ls = []
        with open(file_json) as f:
            data = json.load(f)["data"]
            for item in data:
                paragraphs = item["paragraphs"]
                for paragraph in paragraphs:
                    qas = paragraph["qas"]
                    context = paragraph["context"]
                    for qa in qas:
                        id = qa["id"]
                        question = qa["question"]
                        if not qa["is_impossible"]:
                            if len(qa["answers"]) > 1:
                                print(qa['answers'])
                            answer_start = qa["answers"][0]["answer_start"]
                            ans = qa["answers"][0]["text"]
                        else:
                            answer_start = qa["plausible_answers"][0]["answer_start"]
                            ans = qa["plausible_answers"][0]["text"]
                        self.examples.append(example(id,question,answer_start,ans,qa["is_impossible"],context))
    def __getitem__(self, index):
        ex = self.examples[index]
        ex.question = ex.question.lstrip()
        tokenized_examples = self.tokenizer(ex.question, \
            ex.context,truncation="only_second",max_length = self.max_length, \
                stride =self.doc_stride,return_overflowing_tokens = True,
                return_offsets_mapping=True,padding="max_length")

        offset_mapping = tokenized_examples.pop("offset_mapping")
        tokened_ques = self.tokenizer(ex.question)
        tokenized_examples["id"] = []
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["masked_pos"] = []
        tokenized_examples["masked_tokens"] = []
        tokenized_examples["has_answer"] = []
        tokenized_examples["question_ids"] = tokened_ques["input_ids"]
        tokenized_examples["question_masks"] = tokened_ques["attention_mask"]
        tokenized_examples["question_token_type"] = tokened_ques["token_type_ids"]
        tokenized_examples["answer_text"] = []
        tokenized_examples["masked_input_ids"] = tokenized_examples["input_ids"]
        for i, offsets in enumerate(offset_mapping):
            tokenized_examples["id"].append(ex.id+"_"+str(i))
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            if len(ex.ans) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = ex.start_pos
                end_char = start_char + len(ex.ans)

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["masked_pos"].append([0]*self.max_masked_pred)
                tokenized_examples["masked_tokens"].append([0]*self.max_masked_pred)
                tokenized_examples["has_answer"].append(0)
                tokenized_examples["answer_text"].append("")
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
                # print(self.tokenizer.decode(tokenized_examples["input_ids"][i][token_start_index-1: token_end_index+2]))
                token_start_pos = token_start_index-1
                token_end_pos =  token_end_index+1
                len_cand_mask = token_end_index - token_start_pos
                n_pred =  min(self.max_masked_pred, max(1, int(round(len_cand_mask * 0.15))))
                cand_maked_pos = [pos for pos in range(token_start_pos,token_end_pos+1)]
                # for pos in cand_maked_pos:
                #     print(self.tokenizer.convert_ids_to_tokens(input_ids[pos]))
                shuffle(cand_maked_pos)
                masked_pos = []
                masked_tokens = []
                for pos in cand_maked_pos[:n_pred]:
                    masked_pos.append(pos)
                    masked_tokens.append(input_ids[pos])
                    if random() < 0.8:
                        tokenized_examples["masked_input_ids"] [i][pos] = self.tokenizer.convert_tokens_to_ids('[MASK]') # make mask
                    elif random() < 0.1:
                        index = randint(0, self.tokenizer.vocab_size - 1) # random index in vocabulary
                        inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
                        tokenized_examples["masked_input_ids"][i][pos] = self.tokenizer.convert_tokens_to_ids(inv_vocab[index]) # replace
                if self.max_masked_pred > n_pred:
                    n_pad = self.max_masked_pred - n_pred
                    masked_pos.extend([0] * n_pad)
                    masked_tokens.extend([0] * n_pad)
                tokenized_examples["masked_pos"].append(masked_pos)
                tokenized_examples["masked_tokens"].append(masked_tokens)
                tokenized_examples["has_answer"].append(int(not ex.is_impossible))
                tokenized_examples["answer_text"].append(ex.ans)
        return tokenized_examples

class Input_feature:
    def __init__(self,id,ans,input_ids,masked_input_ids,question_ids,start_position,end_position,attention_mask,token_type_ids,has_answer,masked_tokens,masked_pos,question_masks,question_token_type):
        self.id = id
        self.ans = ans
        self.start_position = start_position
        self.end_position = end_position
        self.has_answer = has_answer
        self.input_ids = input_ids
        self.masked_input_ids = masked_input_ids
        self.masked_pos = masked_pos
        self.masked_tokens = masked_tokens
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.question_id = question_ids
        self.question_mask = question_masks
        self.question_token_type = question_token_type
def convert_to_input_feature(ex,input_feature_list):
        len_features = len(ex["overflow_to_sample_mapping"])
        for i in range(len_features):
            input_ids = ex["input_ids"][i]
            masked_input_ids = ex["masked_input_ids"][i]
            ans = ex["answer_text"][i]
            token_type_ids = ex["token_type_ids"][i]
            attention_mask = ex["attention_mask"][i]
            start_pos = ex["start_positions"][i]
            end_pos = ex["end_positions"][i]
            masked_pos = ex["masked_pos"][i]
            masked_tokens = ex["masked_tokens"][i]
            has_answer = ex["has_answer"][i]
            question_id = ex["question_ids"]
            question_mask = ex["question_masks"]
            question_token_type = ex["question_token_type"]
            id = ex["id"][i]
            input_feature_list.append(Input_feature(id,ans,input_ids,masked_input_ids,question_id,start_pos,end_pos,attention_mask, \
                token_type_ids,has_answer,masked_tokens,masked_pos,question_mask,question_token_type))
        return input_feature_list
        
if __name__ == "__main__":
    tok_ex = QA_data(128,384,"train.json",max_masked_pred=20)[2540]

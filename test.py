import json
from transformers import AutoTokenizer
import torch
from model import QA_Model
from testdataloader import TestDataloader
from tqdm import tqdm

class test_example:
  def __init__(self,id,question,context):
    self.id = id
    self.question = question
    self.context = context

class InputFeature:
  def __init__(self,id,input_ids,token_types,mask_attention,question,context,offset_mapping,question_ids,question_masks,question_token_type):
    self.id = id
    self.input_ids = input_ids
    self.token_types = token_types
    self.mask_attention = mask_attention
    self.question = question
    self.context = context
    self.offset_mapping = offset_mapping
    self.question_id = question_ids
    self.question_mask = question_masks
    self.question_token_type = question_token_type


if __name__ == "__main__":
  tokenizer = AutoTokenizer.from_pretrained("FPTAI/velectra-base-discriminator-cased")

  examples = []
  with open("public_test.json") as test_json:
      test = json.load(test_json)
      data = test["data"]
      for item in data:
          paragraphs = item["paragraphs"]
          for para in paragraphs:
              qas = para["qas"]
              context = para["context"]
              for qa in para["qas"]:
                  question = qa["question"]
                  id = qa["id"]
                  examples.append(test_example(id, question, context))

  input_feature_list = []
  for ex in examples:
      ex.question = ex.question.lstrip()
      tokened_ques = tokenizer(ex.question)
      tokenized_examples = tokenizer(ex.question, \
                                     ex.context, truncation="only_second", max_length=384, \
                                     stride=128, return_overflowing_tokens=True,
                                     return_offsets_mapping=True, padding="max_length")
      tokenized_examples["question_ids"] = tokened_ques["input_ids"]
      tokenized_examples["question_masks"] = tokened_ques["attention_mask"]
      tokenized_examples["question_token_type"] = tokened_ques["token_type_ids"]
      tokenized_examples["id"] = []
      offset_mapping = tokenized_examples["offset_mapping"]
      for i, offsets in enumerate(offset_mapping):
          tokenized_examples["id"].append(ex.id + "_" + str(i))
          input_feature_list.append(InputFeature(tokenized_examples["id"][i], tokenized_examples["input_ids"][i], \
                                                 tokenized_examples["token_type_ids"][i],
                                                 tokenized_examples["attention_mask"][i], \
                                                 ex.question, ex.context, tokenized_examples["offset_mapping"][i],
                                                 tokenized_examples["question_ids"],
                                                 tokenized_examples["question_masks"],
                                                 tokenized_examples["question_token_type"]))

  pred_list = []
  test_dataloader = TestDataloader(input_feature_list, shuffle=False, device="cuda", batch_size=1)
  model = QA_Model().to("cuda")
  model.load_state_dict(torch.load("./saved_model/checkpoint_second.pt")["model"])
  model.eval()
  for id, batch in tqdm(enumerate(test_dataloader)):
      with torch.no_grad():
        pred_list = model(batch, train=False, pred_list=pred_list, tokenizer=tokenizer,input_feature_list=input_feature_list)
  from collections import defaultdict
  submission = defaultdict()
  for pred in pred_list:
      submission[pred.id] = {"ans":pred.pred,"na_score":pred.na_score,
                             "has_score":pred.has_score,
                             "score_verifier":pred.score_verifier}
  submission_file = "results.json"
  open_file = open(submission_file,"w",encoding='utf8')
  json.dump(submission,open_file, indent=6,ensure_ascii=False)
  open_file.close()





class Prediction():
  def __init__(self, id, pred, na_score=0, has_score=0, score_verifier=0):
      self.id = id
      self.pred = pred
      self.na_score = na_score
      self.has_score = has_score
      self.score_verifier = score_verifier

def find_context_question_for_batch(input_feature_list,ids):
    questions = []
    contexts = []
    offset_mappings = []
    for id in ids:
        for input_feature in input_feature_list:
            if id == input_feature.id:
                questions.append(input_feature.question)
                contexts.append(input_feature.context)
                offset_mappings.append(input_feature.offset_mapping)
                break
    return questions,contexts,offset_mappings

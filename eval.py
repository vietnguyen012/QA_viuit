from transformers import ElectraForPreTraining, ElectraTokenizer, AdamW, get_linear_schedule_with_warmup
import pickle
from torch.utils.data import DataLoader
import torch
from apex import amp
import numpy as np
from train import Model, TextDataLoader, TextDataLoader2
import json

tokenizer = ElectraTokenizer.from_pretrained("FPTAI/velectra-base-discriminator-cased", do_lower_case=False)

def normalize(s):
    s = s.replace("&amp;", "&")
    return " ".join(s.split())

def normalize2(s, k=4):
    s = " " + s + " "
    s = s.replace("[", "(")
    s = s.replace("]", ")")
    s = s.replace("{", "(")
    s = s.replace("}", ")")
    s = s.replace("“", '"')
    s = s.replace("”", '"')
    s = s.replace("’", "'")
    s = s.replace(":", " : ")
    s = s.replace(";", " ; ")
    s = s.replace("(", " ( ")
    s = s.replace(")", " ) ")
    s = s.replace(". ", " . ")
    s = s.replace("?", " ? ")
    s = s.replace("!", " ! ")
    s = s.replace("- ", " - ")
    s = s.replace(", ", " , ")
    s = s.replace("=", " = ")
    s = s.replace("*", " * ")
    s = s.replace("/", " / ")
    s = s.replace("',", "' , ")
    if k > 0:
        return normalize2(s, k-1)
    else:
        s = " ".join(s.split())
        s = s.replace("( SEP )", "[SEP]")
        return s

def setup_examples(texts):
    input_ids = []
    label_ids = []
    for word in texts:
        labels = [0]*5
        input_id = tokenizer.tokenize(word)
        label_id = [labels]
        for _ in range(len(input_id) - 1):
            label_id.append([-100]*5)
        input_ids += input_id
        label_ids += label_id
    input_ids = tokenizer.convert_tokens_to_ids(input_ids)
    return input_ids, label_ids


def get_labels(label_tmp, logits, datatype_id, input_id):
    s_id = 0 if input_id == 0 else 128
    e_id = s_id + 128 if len(label_tmp) == 510 else len(label_tmp)
    if input_id == 0:
        e_id += 128
    
    logits = torch.argmax(logits[datatype_id][0], -1)[1:-1]
    all_labels = []
    for id, (label, label_id) in enumerate(zip(label_tmp, logits)):
        label = label[datatype_id]
        label_id = label_id.item()
        if label != -100 and s_id <= id < e_id:
            all_labels.append(label_id)
    
    return all_labels

def get_positions(all_labels):
    results = []
    for id in range(len(all_labels)):
        if all_labels[id] != 0:
            if len(results) > 0 and results[-1][1] == id:
                results[-1][1] += 1
            else:
                results.append([id, id+1])
    return results

def get_ner_results(texts):
    input_ids, label_ids = setup_examples(texts)
    all_labels = [[], [], [], [], []]
    id = 0
    text_id = 0
    while True:
        text = input_ids[id:id+510]
        label_tmp = label_ids[id:id+510]
        train_loader_ner = TextDataLoader2([[text, label_tmp]], shuffle=True, device="cuda", batch_size=32)
        for batch in train_loader_ner:
            logits = model_ner(*batch, mode="NER", return_loss=False)
            labels_0 = get_labels(label_tmp, logits, 0, id)
            labels_1 = get_labels(label_tmp, logits, 1, id)
            labels_2 = get_labels(label_tmp, logits, 2, id)
            labels_3 = get_labels(label_tmp, logits, 3, id)
            labels_4 = get_labels(label_tmp, logits, 4, id)
                        
            all_labels[0] += labels_0
            all_labels[1] += labels_1
            all_labels[2] += labels_2
            all_labels[3] += labels_3
            all_labels[4] += labels_4
            break
        
        if len(label_tmp) < 510:
            break
        
        id += 128
        text_id += len(labels_0)
    
    results_0 = get_positions(all_labels[0])
    results_1 = get_positions(all_labels[1])
    results_2 = get_positions(all_labels[2])
    results_3 = get_positions(all_labels[3])
    results_4 = get_positions(all_labels[4])
    
    return results_0, results_1, results_2, results_3, results_4
    
def get_qa_results(text1, text2, mode, t1=None, t2=None, player_in=None):
    text1 = tokenizer.convert_tokens_to_ids(text1)
    text2 = tokenizer.convert_tokens_to_ids(text2)
    
    train_loader_qa = TextDataLoader([[text1, text2, -1, -1]], shuffle=True, device="cuda", batch_size=32)
    for batch in train_loader_qa:
        s_logits, e_logits = model_qa(*batch, mode="QA", return_loss=False)
        if mode == 0:
            team1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t1))
            team2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t2))
            
            pos = len(text1)+2
            
            for id in range(0, len(text2)):
                if text2[id:id+len(team1)] == team1:
                    s_q_id_1 = id
                if text2[id:id+len(team2)] == team2:
                    s_q_id_2 = id
            
            s_value_1 = torch.max(s_logits[0,pos+s_q_id_1:pos+s_q_id_1+len(team1)]).item()
            s_value_2 = torch.max(s_logits[0,pos+s_q_id_2:pos+s_q_id_2+len(team2)]).item()
            if s_value_1 > s_value_2:
                return t1
            else:
                return t2
        else:
            if player_in is not None:
                player_in = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(player_in))
                for id in range(0, len(text1)):
                    if text1[id:id+len(player_in)] == player_in:
                        s_logits[0,id:id+len(player_in)]=-100
                        e_logits[0,id:id+len(player_in)]=-100
                
            s_id = torch.argmax(s_logits[0,:len(text1)+1]).item()
            e_id = s_id + torch.argmax(e_logits[0,s_id:s_id+10]).item()
            return s_id, e_id

def gen_qa_ids(text_ids, question, c_id):
    text2 = []
    for text in question.split():
        text2 += tokenizer.tokenize(text)
    
    ques_len = len(text2)
    text_len = 510 - ques_len - 1
    pad_len = max(c_id - text_len//2, 0)
    
    text1 = text_ids[pad_len:pad_len+text_len]
    
    return text1, text2

def compare_times(time1, time2):
    if len(time1) == 0 or len(time2) == 0:
        return False
    if time1 == time2:
        return True
    pairs = [("nhất", "1"), ("một", "1"), ("hai", "2"), ("ba", "3"), ("bốn", "4")]
    for x, y in pairs:
        time1 = time1.replace(x, y)
        time2 = time2.replace(x, y)
        if time1 == time2:
            return True
    return False

def _convert_to_str(s, text, c):
    s = " ".join(s.split())
    if s in text: return s
    s1 = s.replace(c + " ", c)
    if s1 in text: return s1
    s2 = s.replace(" " + c, c)
    if s2 in text: return s2
    return s.replace(" " + c + " ", c)

def convert_to_str(tokens, texts_raw, description_raw):
    text = texts_raw + " " + description_raw
    s = tokenizer.convert_tokens_to_string(tokens.split())
    
    while True:
        if any(s.endswith(c) for c in ".,+-/\\'\""):
            s = s[:-1].strip()
        elif any(s.startswith(c) for c in ".,+-/\\'\""):
            s = s[1:].strip()
        else:
            break
    
    s = " " + s + " "
    s = s.split("[SEP]",1)[0]
    s = s.split(",",1)[0]
    s = s.split(" \"",1)[0]
    s = s.split("và ",1)[0]
    for c in ".,+-/\\'\"":
        s = _convert_to_str(s, text, c)
    s = " ".join(s.split())
    if "Chưa tìm được" in s:
        return ""
    return s

def normalize_team(s):
    s = s.replace(".", "")
    s = s.replace("-", "")
    s = " ".join(s.split())
    return s

def check_contain_team(team, clus):
    is_contain = False
    _team = team.lower()
    _clus = [x.lower() for x in clus]
    if _team in _clus:
        is_contain = True
    if any(_team in x for x in _clus):
        is_contain = True
    if normalize_team(_team) in [normalize_team(x) for x in _clus]:
        is_contain = True
    for prefix in ["city", "fc", "united"]:
        if _team + " " + prefix in [x for x in _clus]:
            is_contain = True
            break
        if _team in [x + " " + prefix for x in _clus]:
            is_contain = True
            break
    for prefix in ["st", "st."]:
        if prefix + " " + _team in [x for x in _clus]:
            is_contain = True
            break
        if _team in [prefix + " " + x for x in _clus]:
            is_contain = True
            break
    return is_contain
    
    
def get_match_summary(data):
    original_doc = data["original_doc"]["_source"]

    description_raw = "Mô tả: " + original_doc["description"]
    
    texts = [normalize(doc["text"]) for doc in original_doc["body"]]
    texts_raw = " [SEP] ".join(texts)
    
    texts = normalize2(texts_raw).split()
    descp = normalize2(description_raw).split()
    
    results_ner_descp = get_ner_results(descp)
    results_ner_text = get_ner_results(texts)
    
    all_teams = []
    
    for s_id, e_id in results_ner_descp[1]:
        while descp[s_id] == "[SEP]":
            s_id += 1
        while descp[e_id-1] == "[SEP]":
            e_id -= 1
        if 0 <= s_id < e_id:
            all_teams.append(" ".join(descp[s_id:e_id]))
            all_teams.append(" ".join(descp[s_id:e_id]))
    
    for k, (s_id, e_id) in enumerate(results_ner_text[1]):
        while texts[s_id] == "[SEP]":
            s_id += 1
        while texts[e_id-1] == "[SEP]":
            e_id -= 1
        if 0 <= s_id < e_id:
            all_teams.append(" ".join(texts[s_id:e_id]))
            if k <= 1 or k >= len(results_ner_text[1])-2:
                all_teams.append(" ".join(texts[s_id:e_id]))
                all_teams.append(" ".join(texts[s_id:e_id]))
    
    team_names = [
        ["Barcelona", "Barca", "BARCELONA"],
        ["Man United", "M.U", "MU", "Man Utd", "United", "Manchester United"],
        ["B.Bình Dương", "Bình Dương"],
        ["AC Milan", "Milan"],
        ["Sociedad", "Real Sociedad"], 
        ["SHB.Đà Nẵng", "SHB Đà Nẵng"],
        ["Than Quảng Ninh", "Than.QN"],
        ["Bayern Munich", "Bayern"],
        ["Bayer Leverkusen", "Leverkusen"],
        ["Inter Milan", "Inter"],
        ["Real Madrid", "Real", "REAL MADRID"],
        ["Atletico Madrid", "Atletico"],
        ["Stoke City", "Stoke"],
        ["LA GALAXY", "LA Galaxy"],
        ["Manchester City", "Man City"],
        ["U15 SLNA", "SLNA", "Sông Lam Nghệ An"],
        ["U15 HAGL", "HAGL", "Hoàng Anh Gia Lai"],
        ["Mokpo City FC", "CLB Mokpo"],
        ["Athletic Bilbao", "Athletic"],
        ["U22 VN", "U22 Việt Nam", "Việt Nam", "U22 ViệtNam", "U-22 Việt Nam", "U-22 VN"],
        ["U20 VN", "U20 Việt Nam", "Việt Nam", "U20 ViệtNam", "U-20 Việt Nam", "U-20 VN"],
        ["U.22 Lào", "U22 Lào"],
        ["U22 Campuchia", "Campuchia"],
        ["Futsal Thái Lan", "Thái Lan", "futsal Thái Lan"],
        ["U23 Manchester City", "U23 Man City"],
        ["U23 Arsenal", "Arsenal"],
        ["U22 Thái Lan", "Thái Lan", "U 22 Thái Lan"],
        ["Tuyển nữ Việt Nam", "đội tuyển Việt Nam", "đội tuyển nữ Việt Nam", "tuyển nữ Việt Nam", "tuyển Việt Nam", "ĐT nữ Việt Nam", "nữ Việt Nam", "VN", "Việt Nam", "bóng đá nữ Việt Nam"],
        ["nữ futsal Việt Nam", "futsal nữ Việt Nam", "Việt Nam"],
        ["Burton Albion", "Burton"],
        ["Chelsea", "The Blues"],
        ["FLC Thanh Hóa", "FLC THANH HÓA"],
        ["Olympic Bahrain", "Bahrain"],
        ["RB Leipzig", "Leipzig"],
        ["Bali United", "Bali Utd"],
        ["Argentina", "Agentina"],
        ['Tuyển Việt Nam', 'VN', 'tuyển Việt Nam', 'Việt Nam', 'ĐT Việt Nam'],
        ["Saint-Etienne", "St.Etienne", "Etienne", "St Etienne", "Saint Etienne", "St Ettiene"],
        ["S . Khánh Hòa", "S.Khánh Hòa", "Khánh Hòa"]
    ]
    all_teams.sort(key=lambda x: len(x), reverse=True)
    
    clusters = []
    for team in all_teams:
        is_contain = False
        for clus in clusters:
            if check_contain_team(team, clus):
                is_contain = True
            if not is_contain:
                for names in team_names:
                    names = [x.lower() for x in names]
                    if team.lower() in names:
                        _names = [x for x in names if x != team.lower()]
                        for _name in _names:
                            if check_contain_team(_name, clus):
                                is_contain = True
                                break
                    if is_contain:
                        break
                    
            if is_contain:
                clus.append(team)
                break
        if not is_contain:
            clusters.append([team])
    
    clusters.sort(key=lambda x: len(x), reverse=True)
    
    if len(clusters) > 0:
        print("clusters:")
        for x in clusters:
            print("  -", list(set(x)))
    
    if len(clusters) == 1:
        clusters.append(clusters[0])
    
    teams1 = list(set(clusters[0]))
    teams1.sort(key=clusters[0].count, reverse=True)
    teams2 = list(set(clusters[1]))
    teams2.sort(key=clusters[1].count, reverse=True)
    team1 = teams1[0]
    team2 = teams2[0]
    
    if team1 in team2 or team2 in team1:
        if team1 in team2:
            team1 = team2
        teams = [x for x in all_teams if x.lower() not in team1.lower()]
        _teams = list(set(teams))
        _teams.sort(key=all_teams.count, reverse=True)
        if len(_teams) > 0:
            team2 = _teams[0]
        else:
            team2 = all_teams[0]
        
    print("team1:", team1, "-", team2)
    
    """
    teams1 = []
    teams2 = []
    for team in clusters[0]:
        for names in team_names:
            names = [x.lower() for x in names]
            if team.lower() in names:
                teams1 += names
    for team in clusters[1]:
        for names in team_names:
            names = [x.lower() for x in names]
            if team.lower() in names:
                teams2 += names
    teams1 = list(set(teams1))
    teams1.sort(key=lambda x: len(x), reverse=True)
    teams2 = list(set(teams2))
    teams2.sort(key=lambda x: len(x), reverse=True)
    """
                
    all_scores = []
    
    for s_id, e_id in results_ner_descp[0]:
        while descp[s_id] == "[SEP]":
            s_id += 1
        while descp[e_id-1] == "[SEP]":
            e_id -= 1
        if 0 <= s_id < e_id:
            all_scores.append(" ".join(descp[s_id:e_id]))
            all_scores.append(" ".join(descp[s_id:e_id]))
            all_scores.append(" ".join(descp[s_id:e_id]))
            
    for k, (s_id, e_id) in enumerate(results_ner_text[0]):
        while texts[s_id] == "[SEP]":
            s_id += 1
        while texts[e_id-1] == "[SEP]":
            e_id -= 1
        if 0 <= s_id < e_id:
            all_scores.append(" ".join(texts[s_id:e_id]))
            if k == 0 or k == len(results_ner_text[0])-1:
                all_scores.append(" ".join(texts[s_id:e_id]))
                all_scores.append(" ".join(texts[s_id:e_id]))
    
    if len(all_scores) > 0:
        #print("all_scores:", all_scores)
        scores = []
        for score in all_scores:
            parts = score.split("-",1)
            parts.sort(reverse=True)
            if len(parts) != 2:
                continue
            parts[0] = parts[0].strip()
            parts[1] = parts[1].strip()
            scores.append("-".join(parts))
        _scores = list(set(scores))
        _scores.sort(key=scores.count, reverse=True)
        #print("_scores:", _scores)
        if len(_scores) > 0:
            score = _scores[0]
            score = score.split("-")
        
    text_ids = []
    id_mapping = {}
    for k, text in enumerate(texts):
        id_mapping[k] = len(text_ids)
        text_ids += tokenizer.tokenize(text)
        
    scores_info = []
    for s_id, e_id in results_ner_text[2]:
        player_name = " ".join(texts[s_id:e_id])
        
        c_id = (s_id + e_id)//2
        c_id = id_mapping[c_id]
        
        question = f"Cầu thủ {player_name} thuộc đội bóng {team1} hay {team2} ?"
        text1, text2 = gen_qa_ids(text_ids, question, c_id)
        text_qa = ["[SEP]"] + text1 + ["[SEP]"] + text2
        team = get_qa_results(text1, text2, 0, team1, team2)
        """
        teams = teams1 if team == team1 else teams2
        texts_raw_lower = " ".join(texts[s_id-256:s_id+128]).lower()
        for t in teams:
            t = normalize2(t)
            if t in texts_raw_lower:
                k = len(texts_raw_lower.split(t)[0].split())
                t = " ".join(texts[s_id-256+k:s_id-256+k+len(t.split())])
                team = t
                break
        """
        
        question = f"Cầu thủ {player_name} ghi bàn lúc nào ?"
        text1, text2 = gen_qa_ids(text_ids, question, c_id)
        text_qa = ["[SEP]"] + text1 + ["[SEP]"] + text2
        s_id, e_id = get_qa_results(text1, text2, 1)
        time = " ".join(text_qa[s_id:e_id])
        while time.endswith("'") or time.endswith('"'):
            time = time[:-1].strip()
        team = convert_to_str(team, texts_raw, description_raw)
        time = convert_to_str(time, texts_raw, description_raw)
        player_name = convert_to_str(player_name, texts_raw, description_raw)
        scores_info.append((player_name, team, time))
    
    final_score_infor = []
    for player_name, team, time in scores_info:
        if len(final_score_infor) == 0 or all(not compare_times(time, x[-1]) for x in final_score_infor):
            final_score_infor.append((player_name, team, time))
        
    cards_info = []
    for s_id, e_id in results_ner_text[3]:
        player_name = " ".join(texts[s_id:e_id])
        
        c_id = (s_id + e_id)//2
        c_id = id_mapping[c_id]
        
        question = f"Cầu thủ {player_name} thuộc đội bóng {team1} hay {team2} ?"
        text1, text2 = gen_qa_ids(text_ids, question, c_id)
        text_qa = ["[SEP]"] + text1 + ["[SEP]"] + text2
        team = get_qa_results(text1, text2, 0, team1, team2)
        """
        teams = teams1 if team == team1 else teams2
        texts_raw_lower = " ".join(texts[s_id-256:s_id+128]).lower()
        for t in teams:
            t = normalize2(t)
            if t in texts_raw_lower:
                k = len(texts_raw_lower.split(t)[0].split())
                t = " ".join(texts[s_id-256+k:s_id-256+k+len(t.split())])
                team = t
                break
        """
        question = f"Cầu thủ {player_name} bị phạt lúc nào ?"
        text1, text2 = gen_qa_ids(text_ids, question, c_id)
        text_qa = ["[SEP]"] + text1 + ["[SEP]"] + text2
        s_id, e_id = get_qa_results(text1, text2, 1)
        time = " ".join(text_qa[s_id:e_id])
        while time.endswith("'") or time.endswith('"'):
            time = time[:-1].strip()
        team = convert_to_str(team, texts_raw, description_raw)
        time = convert_to_str(time, texts_raw, description_raw)
        player_name = convert_to_str(player_name, texts_raw, description_raw)
        cards_info.append((player_name, team, time))
    
    substitutions_info = []
    for s_id, e_id in results_ner_text[3]:
        player_in = " ".join(texts[s_id:e_id])
        c_id = (s_id + e_id)//2
        c_id = id_mapping[c_id]
        
        question = f"{player_in} vào sân thay thế cho cầu thủ nào ?"
        text1, text2 = gen_qa_ids(text_ids, question, c_id)
        text_qa = ["[SEP]"] + text1 + ["[SEP]"] + text2
        s_id, e_id = get_qa_results(text1, text2, 1, player_in=player_in)
        player_out = " ".join(text_qa[s_id:e_id])
        
        question = f"Cầu thủ {player_in} vào sân lúc nào ?"
        text1, text2 = gen_qa_ids(text_ids, question, c_id)
        text_qa = ["[SEP]"] + text1 + ["[SEP]"] + text2
        s_id, e_id = get_qa_results(text1, text2, 1)
        time = " ".join(text_qa[s_id:e_id])
        while time.endswith("'") or time.endswith('"'):
            time = time[:-1].strip()
        
        player_in = convert_to_str(player_in, texts_raw, description_raw)
        player_out = convert_to_str(player_out, texts_raw, description_raw)
        if any(x.isdigit() for x in player_out) or any(x[0].islower() for x in player_out.split()) or len(player_out.split()) >= 5:
            player_out = ""
        time = convert_to_str(time, texts_raw, description_raw)
        substitutions_info.append((player_in, player_out, time))
    
    final_substitutions_info = []
    for player_in, player_out, time in substitutions_info:
        if len(final_substitutions_info) == 0 or all(player_in not in x[0] and x[0] not in player_in for x in final_substitutions_info):
            if len(final_substitutions_info) > 0 and any(player_out in x[1] or (x[1] in player_out and len(x[1]) > 0) for x in final_substitutions_info):
                player_out = ""
            final_substitutions_info.append((player_in, player_out, time))
    
    
    score1 = len([x for x in final_score_infor if x[1] == team1])
    score2 = len([x for x in final_score_infor if x[1] == team2])
    
    try:
        s1 = int(score[0].strip())
        s2 = int(score[1].strip())
        if score1 > score2:
            if s1 > s2:
                score1 = s1
                score2 = s2
            else:
                score1 = s2
                score2 = s1
        else:
            if s1 < s2:
                score1 = s1
                score2 = s2
            else:
                score1 = s2
                score2 = s1
    except:
        pass
    
    
    
    score_list = [{"player_name": x[0], "team": x[1], "time": x[2]} for x in final_score_infor]
    card_list = [{"player_name": x[0], "team": x[1], "time": x[2]} for x in cards_info]
    substitution_list = [{"player_in": x[0], "player_out": x[1], "time": x[2]} for x in final_substitutions_info]
    
    match_summary = {
        "players": {
            "team1": team1,
            "team2": team2
        },
        "score_board": {
            "score1": str(score1),
            "score2": str(score2)
        },
        "score_list": score_list,
        "card_list": card_list,
        "substitution_list": substitution_list
    }
    
    results = {
        "test_id": data["test_id"],
        "match_summary": match_summary
    }
    return results

model_ner = Model()
model_ner.load_state_dict(torch.load("checkpoints/model_130298_ner.pkl"), strict=False)
model_ner.eval()

model_qa = Model()
model_qa.load_state_dict(torch.load("checkpoints/model_130298_qa.pkl"), strict=False)
model_qa.eval() 
    
if __name__ == "__main__":
    data = {"test_id": "23583960", "original_doc": {"_id": "23583960", "_source": {"body": [{"content": "Inter Milan không thay đổi quá nhiều nhưng điều quan trọng là họ tìm được sự kết dính nhờ tài năng của HLV Luciano Spaletti. Không chỉ gắn kết, Nerazzurri đang sở hữu ngôi sao biết ghi dấu ấn ở những trận đấu lớn, đó là Mauri Icardi, người đã ghi cả 3 bàn thắng vào lưới Milan đêm qua.", "text": "Inter Milan không thay đổi quá nhiều nhưng điều quan trọng là họ tìm được sự kết dính nhờ tài năng của HLV Luciano Spaletti. Không chỉ gắn kết, Nerazzurri đang sở hữu ngôi sao biết ghi dấu ấn ở những trận đấu lớn, đó là Mauri Icardi, người đã ghi cả 3 bàn thắng vào lưới Milan đêm qua.", "type": "text"}, {"content": "Dưới thời HLV Spaletti, Inter khá thành công với sơ đồ 4-2-3-1 với trung phong cắm Mauri Icardi. Công thức này tiếp tục được cựu HLV Roma sử dụng ở trận derby Milan đêm qua. Trong khi đó, Milan lại ra sân với sơ đồ 3-5-2. Do thiếu vắng 1 số nhân tố quan trọng như Conti, Kalinic hay Calhanoglu nên Milan nhập cuộc không tốt.", "text": "Dưới thời HLV Spaletti, Inter khá thành công với sơ đồ 4-2-3-1 với trung phong cắm Mauri Icardi. Công thức này tiếp tục được cựu HLV Roma sử dụng ở trận derby Milan đêm qua. Trong khi đó, Milan lại ra sân với sơ đồ 3-5-2. Do thiếu vắng 1 số nhân tố quan trọng như Conti, Kalinic hay Calhanoglu nên Milan nhập cuộc không tốt.", "type": "text"}, {"content": "Nói đúng hơn, sự khác biệt lớn nhất đến từ sự kết dính giữa các tuyến của 2 đội, trong đó Inter Milan đã thể hiện được sự gắn kết giữa các tuyến và bảo đảm 1 thế trận lấn lướt trước đối thủ cùng thành phố. Ngay ở phút 13, Nerazzurri đã đe dọa khung thành Milan với cú sút đưa bóng dội xà ngang của Antonio Candreva.", "text": "Nói đúng hơn, sự khác biệt lớn nhất đến từ sự kết dính giữa các tuyến của 2 đội, trong đó Inter Milan đã thể hiện được sự gắn kết giữa các tuyến và bảo đảm 1 thế trận lấn lướt trước đối thủ cùng thành phố. Ngay ở phút 13, Nerazzurri đã đe dọa khung thành Milan với cú sút đưa bóng dội xà ngang của Antonio Candreva.", "type": "text"}, {"content": "Không ghi bàn ở tình huống này nhưng Candreva đã ghi dấu ấn ở phút 28 với cú căng ngang như đặt để Mauro Icardi cắt mặt ghi bàn mở tỷ số cho Inter Milan.", "text": "Không ghi bàn ở tình huống này nhưng Candreva đã ghi dấu ấn ở phút 28 với cú căng ngang như đặt để Mauro Icardi cắt mặt ghi bàn mở tỷ số cho Inter Milan.", "type": "text"}, {"content": "Bàn mở tỷ số của tiền đạo người Argentina đã khiến trận đấu trở nên hấp dẫn hơn bởi đó cũng là thời điểm Milan buộc phải dâng cao đội hình tấn công.", "text": "Bàn mở tỷ số của tiền đạo người Argentina đã khiến trận đấu trở nên hấp dẫn hơn bởi đó cũng là thời điểm Milan buộc phải dâng cao đội hình tấn công.", "type": "text"}, {"content": "Áp lực mà Milan tạo ra thực sự gây khó cho Inter Milan ở đầu hiệp 2. Sau liên tiếp các tình huống nguy hiểm tạo ra, cuối cùng Rossoneri đã có bàn gỡ hòa 1-1 ở phút 56 nhờ cú đặt lòng hiểm hóc của Suso ở ngoài vòng cấm.", "text": "Áp lực mà Milan tạo ra thực sự gây khó cho Inter Milan ở đầu hiệp 2. Sau liên tiếp các tình huống nguy hiểm tạo ra, cuối cùng Rossoneri đã có bàn gỡ hòa 1-1 ở phút 56 nhờ cú đặt lòng hiểm hóc của Suso ở ngoài vòng cấm.", "type": "text"}, {"content": "Hưng phấn sau bàn gỡ hòa, Milan vùng lên tấn công dữ dội. Nhưng ở thời điểm đó, họ lại phải nếm trái đắng từ lối chơi phản công rất khó chịu của Inter Milan. Một lần nữa Mauro Icardi lại được hô vang ở Giuseppe Meazza với cú dứt điểm bóng sống sau pha phản công chớp nhoáng.", "text": "Hưng phấn sau bàn gỡ hòa, Milan vùng lên tấn công dữ dội. Nhưng ở thời điểm đó, họ lại phải nếm trái đắng từ lối chơi phản công rất khó chịu của Inter Milan. Một lần nữa Mauro Icardi lại được hô vang ở Giuseppe Meazza với cú dứt điểm bóng sống sau pha phản công chớp nhoáng.", "type": "text"}, {"content": "Sau bàn thắng vượt lên dẫn trước, Inter Milan đã chủ động chơi phòng ngự. Nhưng trước áp lực rất lớn mà Milan tạo ra, Inter đã bị gỡ hòa ở phút 81 nhờ cú dứt điểm của Bonaventura. Thủ thành Handanovic đã cản phá nhưng trọng tài xác định bóng đã lăn qua vạch cầu môn.", "text": "Sau bàn thắng vượt lên dẫn trước, Inter Milan đã chủ động chơi phòng ngự. Nhưng trước áp lực rất lớn mà Milan tạo ra, Inter đã bị gỡ hòa ở phút 81 nhờ cú dứt điểm của Bonaventura. Thủ thành Handanovic đã cản phá nhưng trọng tài xác định bóng đã lăn qua vạch cầu môn.", "type": "text"}, {"content": "Những tưởng trận đấu kết thúc với tỷ số hòa 2-2 thì đúng vào phút thi đấu chính thức cuối cùng, hậu vệ Ricardo Rodriguez đã mắc sai lầm với pha kéo ngã cầu thủ đối phương trong vòng cấm. Trọng tài cất còi cho Inter Milan được hưởng phạt đền. Trên chấm 11m, Mauro Icardi đã hoàn tất cú hat-trick ấn định chiến thắng 3-2 đầy kịch tính cho Inter Milan.", "text": "Những tưởng trận đấu kết thúc với tỷ số hòa 2-2 thì đúng vào phút thi đấu chính thức cuối cùng, hậu vệ Ricardo Rodriguez đã mắc sai lầm với pha kéo ngã cầu thủ đối phương trong vòng cấm. Trọng tài cất còi cho Inter Milan được hưởng phạt đền. Trên chấm 11m, Mauro Icardi đã hoàn tất cú hat-trick ấn định chiến thắng 3-2 đầy kịch tính cho Inter Milan.", "type": "text"}, {"content": "3 điểm giành được ở derby della Madonnina đã giúp Inter Milan nối dài mạch bất bại ở Serie A mùa này lên con số 8 và tiếp tục duy trì vị trí thứ 2 trên BXH và chỉ kém đội đầu bảng Napoli 2 điểm. Trong khi đó, với thất bại thứ 4 kể từ đầu mùa, Rossoneri sau một mùa hè mua sắm rầm rộ, họ vẫn chưa cải thiện tình hình khi tụt xuống vị trí thứ 10 và nguy cơ HLV Vincenzo Montella bị sa thải là rất cao.", "text": "3 điểm giành được ở derby della Madonnina đã giúp Inter Milan nối dài mạch bất bại ở Serie A mùa này lên con số 8 và tiếp tục duy trì vị trí thứ 2 trên BXH và chỉ kém đội đầu bảng Napoli 2 điểm. Trong khi đó, với thất bại thứ 4 kể từ đầu mùa, Rossoneri sau một mùa hè mua sắm rầm rộ, họ vẫn chưa cải thiện tình hình khi tụt xuống vị trí thứ 10 và nguy cơ HLV Vincenzo Montella bị sa thải là rất cao.", "type": "text"}, {"content": "<strong>Đội hình thi đấu:</strong>", "text": "Đội hình thi đấu:", "type": "text"}, {"content": "Inter Milan (4-2-3-1): Handanovic; D'Ambrosio, Skriniar, Miranda, Nagatomo; Vecino, Gagliardini; Candreva (Cancelo 73), Valero (Eder 85), Perisic; Icardi (Santon 90+2)<br>Subs not used: Ranocchia, Karamoh, Padelli, Henrique, Berni, Pinamonti<br>Ghi bàn: Icardi 28, 63, 90 pen", "text": "Inter Milan (4-2-3-1): Handanovic; D'Ambrosio, Skriniar, Miranda, Nagatomo; Vecino, Gagliardini; Candreva (Cancelo 73), Valero (Eder 85), Perisic; Icardi (Santon 90+2)Subs not used: Ranocchia, Karamoh, Padelli, Henrique, Berni, PinamontiGhi bàn: Icardi 28, 63, 90 pen", "type": "text"}, {"content": "AC Milan: (3-5-2): Donnarumma; Musacchio, Bonucci, Romagnoli (Locatelli 78); Borini, Kessie (Cutrone 45), Biglia, Bonaventura, Rodriguez; Suso, Andre Silva<br>Ghi bàn: Suso 56; Bonaventura 81", "text": "AC Milan: (3-5-2): Donnarumma; Musacchio, Bonucci, Romagnoli (Locatelli 78); Borini, Kessie (Cutrone 45), Biglia, Bonaventura, Rodriguez; Suso, Andre SilvaGhi bàn: Suso 56; Bonaventura 81", "type": "text"}, {"content": "<strong>Như Ý</strong>", "text": "Như Ý", "type": "text"}], "content_id": 23583960, "description": "Đã nhiều năm rồi mới được chứng kiến trận 'derby della Madonnina' kịch tính và hấp dẫn đến vậy, và người mang đến cảm giác vỡ òa cho Inter Milan là Mauro Icardi với cú hat-trick để đánh bại AC Milan 3-2 tại Giuseppe Meazza.", "last_update": "1520451445150", "publish_date": 1508121180}}}
    #results = get_match_summary(data)
    
    submission = []
    with open("dataset/dataset/public_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            results = get_match_summary(data)
            submission.append(results)
            print("results:", results)
            

    with open("submission.jsonl", "w", encoding="utf-8") as f:
        for data in submission:
            f.write(json.dumps(data) + "\n")
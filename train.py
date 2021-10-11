from transformers import AutoTokenizer
from data import QA_data,convert_to_input_feature
from data_loader import QA_Dataloader
from tqdm import tqdm
import torch
from transformers.optimization import get_linear_schedule_with_warmup,AdamW
from random import shuffle
from evaluation import compute_exact,compute_f1
from model import QA_Model
seed= 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from apex import amp
import os

def train(model,train_data_loader,optimizer,scheduler,epoch):
    model.train()
    losses = []
    total_logit_losses=[]
    loss_has=[]
    loop = tqdm(tqdm(train_data_loader),position=0, leave=True)
    for (step, batch) in enumerate(loop):
        loss,total_logit_loss,loss_ha = model(batch)
        losses.append(loss.item())
        total_logit_losses.append(total_logit_loss.item())
        loss_has.append(loss_ha)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        if amp._amp_state.loss_scalers[0]._unskipped != 0:
            scheduler.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        mean_loss = sum(losses) / len(losses)
        mean_total_logits = sum(total_logit_losses)/len(total_logit_losses)
        mean_loss_has = sum(loss_has)/len(loss_has)
        loop.set_postfix(epoch=epoch,loss=mean_loss,mean_total_logits=mean_total_logits,mean_loss_lm_has=mean_loss_has)

def eval(model,val_data_loader,tokenizer,device="cuda",delta=0.5):
    model.eval()
    losses = []
    f1 = []
    em = []
    loop = tqdm(tqdm(val_data_loader),position=0, leave=True)
    model = model.to(device)
    for (step, batch) in enumerate(loop):
        with torch.no_grad():
            start_logits, end_logits, has_ans_logits, loss = model(batch,return_logits=True)
            clf = torch.argmax(torch.nn.Softmax(dim=-1)(has_ans_logits),dim=-1)
            losses.append(loss.item())
            has_answer = False
            for ind,(start_logit,end_logit,is_has) in enumerate(zip(start_logits,end_logits,clf)):
                offset_mapping = batch["offset_mappings"][ind]
                start_pos = torch.argmax(start_logit)
                end_pos = torch.argmax(end_logit)
                if end_pos < start_pos or is_has == 0 or end_pos == start_pos == 0 or \
                        start_pos >= len(offset_mapping) or end_pos >= len(offset_mapping) \
                 or offset_mapping[start_pos] is None or offset_mapping[end_pos] is None:
                    pred_ans = ""
                else:
                    start_char = offset_mapping[start_pos][0]
                    end_char = offset_mapping[end_pos][1]
                    has_answer=True
                found = False
                for input_feature in input_feature_list:
                    if input_feature.id == batch["ids"][ind]:
                        if batch["has_answer"][ind] == 0:
                            gold_ans = ""
                            found = True
                            break
                        else:
                            gold_ans = input_feature.context[offset_mapping[batch["start_position"][ind]][0]:offset_mapping[batch["end_position"][ind]][1]]
                            found = True
                            break
                assert found == True,print("input_feature.id,batch[id]:",input_feature.id,batch["ids"][ind])
                if has_answer:
                    pred_ans = input_feature.context[start_char:end_char]
                em_score = compute_exact(gold_ans,pred_ans)
                f1_score = compute_f1(gold_ans,pred_ans)
                f1.append(f1_score)
                em.append(em_score)
            mean_loss = sum(losses) / len(losses)
            mean_f1 = sum(f1)/len(f1)
            mean_em = sum(em)/len(em)
            loop.set_postfix(val_loss=mean_loss,f1=mean_f1,exact_match=mean_em)
    return mean_f1,mean_em


def get_optimizer(model, bert_lr, lr, bert_weight_decay=0.05, adam_epsilon=1e-8):
    optimizer_grouped_parameters = []
    for n, p in model.named_parameters():
        optimizer_params = {"params": p}
        if "encoder" in n:
            optimizer_params["lr"] = bert_lr
            if any(x in n for x in ['bias', 'LayerNorm.weight']):
                optimizer_params["weight_decay"] = 0
            else:
                optimizer_params["weight_decay"] = bert_weight_decay
        else:
            optimizer_params["lr"] = lr
        optimizer_grouped_parameters.append(optimizer_params)
    return AdamW(optimizer_grouped_parameters, eps=adam_epsilon)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("FPTAI/velectra-base-discriminator-cased", do_lower_case=False)
    examples = QA_data(128, 384, "train.json", max_masked_pred=20)
    input_feature_list = []
    for ex in examples:
        input_feature_list = convert_to_input_feature(ex, input_feature_list)
    shuffle(input_feature_list)
    train_size = int(len(input_feature_list) * 0.8)
    train_input_feature_list, val_input_feature_list = input_feature_list[:train_size], input_feature_list[train_size:]
    use_checkpoint=False
    weight_decay = 0.01
    learning_rate = 2e-3
    adam_epsilon = 1e-6
    num_warmup_steps = 2
    no_decay = ['bias', 'LayerNorm.weight']
    train_examples = len(train_input_feature_list)
    train_batch_size = 40
    val_batch_size = 64
    num_train_epochs = 100
    warmup_proportion = 0.1
    num_train_steps = int( \
        train_examples / train_batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    train_data_loader = QA_Dataloader(train_input_feature_list, shuffle=True, device="cuda", batch_size=train_batch_size)
    val_data_loader = QA_Dataloader(val_input_feature_list, device="cuda", batch_size=val_batch_size)
    model = QA_Model().to("cuda")
    optimizer = get_optimizer(model, 2e-5, 2e-5)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)
    print("Begin training with {} train examples, {} val examples "
                 "{} epochs, {} batches, {} learning_rate".format(train_examples,len(val_input_feature_list),num_train_epochs,train_batch_size,learning_rate))
    besr_f1 = -1
    best_em = -1
    if use_checkpoint:
        checkpoint = torch.load('./saved_model/checkpoint.pt')
        epoch = checkpoint['epoch']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        f1 = checkpoint['best_f1']
        em = checkpoint['best_exact_match']
        print(f"previous f1:{f1}  exact match:{em}")
    print("model:",model.eval())
    print("======================training======================")
    for epoch in range(num_train_epochs):
        print(f"======================epoch={epoch}=========================")
        train(model, train_data_loader, optimizer, scheduler,epoch)
        print("======================validating====================")
        max_f1 = -1
        max_em = -1
        val_delta = 0.5
        if epoch == 0 or epoch%7==0:
            for delta in ([ 0.5]):
                 f1, em = eval(model, val_data_loader, tokenizer, delta=delta)
                 if f1 > max_f1 or em > max_em:
                     max_f1 = f1
                     max_em = em
                     val_delta = delta
                     print(f"update new delta:{val_delta}")
        else:
            f1, em = eval(model, val_data_loader, tokenizer, delta=val_delta)
            if epoch % 9 == 0 and epoch != 0:
                if best_em < em or besr_f1 < f1 :
                    best_em = em
                    best_f1 = f1
                    if not os.path.exists("saved_model"):
                        os.makedirs("saved_model")
                    checkpoint = {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_exact_match':best_em,
                        'best_f1':best_f1,
                        'threshold':val_delta}
                    torch.save(checkpoint, './saved_model/checkpoint.pt')
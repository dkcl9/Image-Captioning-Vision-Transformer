# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:48:55 2024

@author: aidan kim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Transformer
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
import json
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from util.bleu import get_bleu
import numpy as np

#Class to perform early stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='saved_model.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        #Saves model when validation loss decrease.
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss




class Dataset(Dataset):
    def __init__(self, Img,caption,transform=None):
        self.Img = Img
        self.caption = caption
        self.transform = transform
    
    def __len__(self):
        return len(self.Img)
    
    def __getitem__(self, idx):
        Img = self.Img[idx]
        caption = self.caption[idx]
        if self.transform:
            Img = self.transform(Img)
        return (Img,caption)

def create_batch(each_data_batch,PAD_IDX):
     caption_batch = []
     Image_batch = []
     for (Img,caption) in each_data_batch:
         Image_batch.append(Img)
         caption_batch.append(caption)
 
     Image_batch = torch.stack(Image_batch, dim=0)
     caption_batch = pad_sequence(caption_batch, padding_value=PAD_IDX)
     return Image_batch,caption_batch



# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class MyTransformer(nn.Module):
    def __init__(self,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(MyTransformer, self).__init__()
        # Vision Transformer Encoder
        self.v = ViT(
        image_size=256, #Size of the imagepatch_size=16, # number of pathcs
        patch_size=16, # number of pathcs
        num_classes=1000, #number of classes,
        dim=768, # Embedding dimension
        depth=5, # No of encoder layers
        heads=12, #No of MLP heads
        mlp_dim=768, # Dimension of Feedforwards
        dropout=0.2,
        emb_dropout=0.2
        )
        self.vit = Extractor(self.v)
        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_decoder_layers
        )

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src: Tensor, trg: Tensor,PAD_IDX: Tensor):
        # Process input through the vision transformer
        logits, memory = self.vit(src)  # src is expected to be images of shape (batch_size, 3, image_size, image_size)

        #trg_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        tgt_mask = self.generate_square_subsequent_mask(trg.shape[0])
        #tgt_key_padding_mask = (tgt == PAD_IDX).transpose(0, 1)  # PAD_IDX needs to be defined or passed

        # Decode the target sequence
        #outs = self.transformer_decoder(trg_emb, memory, tgt_mask, None, tgt_key_padding_mask)
        outs = self.decode(trg, memory.transpose(0, 1), tgt_mask)
        return self.generator(outs)


    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)





def get_data(json_file_path,img_dir,transform):
    image_data= []
    target = [[],[],[],[],[]]
    json_file = json.load(open(json_file_path))
    for data in json_file['images']:
        item = data
        img_path = os.path.join(img_dir, item['filename'])
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        image_data.append(image)
        for i in range(5):
            tokens = data['sentences'][i]['tokens']
            target[i].append(tokens)
    
    return image_data, target


def make_vocab(tokens):
    specials = ['<unk>', '<pad>','<bos>','<eos>']
    all_tokens = []
    for ids in tokens:
        for sentences in ids:
            for token in sentences:
                all_tokens.append(token)
    unique_tokens = set(all_tokens)
    unique_tokens = specials + sorted(list(unique_tokens))
    vocabulary = {}
    index = 0
    for key in unique_tokens:
        vocabulary[key] = index
        index += 1
    return vocabulary


def tokenize_sentence(sentence, vocab):
    token_arr = [vocab['<bos>']]
    for token in sentence:
        if token in vocab.keys():
            token_arr.append(vocab[token])
        else:
            token_arr.append(vocab['<unk>']) #If given word is not in vocab set it to unknown
    token_arr.append(vocab['<eos>'])
    return token_arr

#tokenize all data sentence using given vocab
def tokenize_all_sentences(target, vocab):
    tokenized_sentences = []
    for ids in target:
        captions = []
        for sentence in ids:
            captions.append(torch.tensor(tokenize_sentence(sentence, vocab)))
        tokenized_sentences .append(captions)
    return tokenized_sentences

img = transforms.Compose([transforms.Resize((256, 256)),
 transforms.ToTensor(),
 transforms.Normalize(0.5, 0.5),
 ])

#Preprocessing datas
img_dir = 'Images/Images'
print("=======Data preprocessing=======")
train_image, train_tokens = get_data('training_data.json','Images/Images',img)
val_image, val_tokens = get_data('val_data.json','Images/Images',img)
test_image, test_tokens = get_data('test_data.json','Images/Images',img)

vocab = make_vocab(train_tokens)

train_target = tokenize_all_sentences(train_tokens,vocab)
val_target = tokenize_all_sentences(val_tokens,vocab)
test_target = tokenize_all_sentences(test_tokens,vocab)
print("=========Finished==========")
print()

#initialize model
NUM_DECODER_LAYERS = 5
EMB_SIZE = 768  # Embedding dimension
NHEAD = 12 # Number of Attention heads
FFN_HID_DIM = 768  # Feedforward dimension
TGT_VOCAB_SIZE = len(vocab)
print()
model = MyTransformer(
NUM_DECODER_LAYERS,
EMB_SIZE, NHEAD,
TGT_VOCAB_SIZE,
FFN_HID_DIM)

PAD_IDX = vocab['<pad>']
optimizer = torch.optim.Adam(
model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


model = model.to(device)



BATCH_SIZE = 50

#image augmentation for training data
train_trsf = transforms.Compose([

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.RandomAutocontrast(p=0.5)

 ])

#function to train one epoch
def train_epoch(model,train_image,train_target,optimizer):
    total_loss = 0
    model.train()
    print('-----training----')
    #iterate all 5 captions
    for i in range(5):
        j = 0
        #train_data = Dataset(train_image,train_target[i],train_trsf)
        train_data = Dataset(train_image,train_target[i])
        train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: create_batch(x, PAD_IDX))
        for src, tgt in train_iter:
            src = src.to(device)
            tgt = tgt.to(device)
            logits = model.forward(src,tgt[0:-1,:],torch.tensor(PAD_IDX))
            optimizer.zero_grad()
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            j += 1
            print("caption:"+str(i+1)+"    "+str(j) + '/120  processed..'+'  loss: '+str(loss.item()),end='\r', flush=True)
    train_loss = total_loss/(len(list(train_iter))*5)
    return train_loss

#function to validate model
def val_epoch(model,val_image,val_target):
    model.eval()
    total_loss = 0
    print('-----evaluating----')
    for i in range(5):
        j = 0
        val_data = Dataset(val_image,val_target[i])
        val_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: create_batch(x, PAD_IDX))
        for src, tgt in val_iter:
            src = src.to(device)
            tgt = tgt.to(device)
            logits = model.forward(src,tgt[0:-1,:],torch.tensor(PAD_IDX))
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            total_loss += loss.item()
            j += 1
            print("caption:"+str(i+1)+"    "+str(j) + '/20  processed..'+'  loss: '+str(loss.item()),end='\r', flush=True)
    val_loss = total_loss/(len(list(val_iter))*5)
    return val_loss

#main train function
def train(model,train_image,val_image,train_target,val_target,optimizer,test_image,test_target):
    epoch = 0
    train_losses = []
    val_losses = []
    bleu_scores = []
    early_stopper = EarlyStopping(patience=3, verbose=True, path='saved_model.pt')
    max_epoch = 15


    for e in range(max_epoch):
        epoch +=1
        print("epoch:",epoch)
        train_loss = train_epoch(model,train_image,train_target,optimizer)
        train_losses.append(train_loss)
        val_loss = val_epoch(model,val_image,val_target)
        val_losses.append(val_loss)
        print("train_loss: {}     val_loss: {}".format(train_loss,val_loss))
        early_stopper(val_loss, model)#check whether validation loss is decreased
        if(val_loss <= 3.1 or epoch == 15):
            torch.save(model.state_dict(), '31.pt')
            bleu = test(test_image,test_target)
            bleu_scores.append(bleu)
        
        
        if early_stopper.early_stop:#If it pass 5epoch without improvement eraly stop the training
            print("Early stopping")
            break
     
    print_loss(train_losses,val_losses)
    print_bleu(bleu_scores,epoch)





def test(test_image,test_target):
    #Get model parameters
    model.eval()
    total_score = 0
    
    test_data = Dataset(test_image,test_target[0])
    w = 0
    for i in range(len(test_data)):
        src,tgt = test_data[i]
        src = src.to(device)
        logits_, input_encoder = model.vit(src.unsqueeze(0))
        input_encoder = input_encoder.to(device)
        pred = torch.tensor([[0]])
        pred[0,0] = vocab['<bos>']
        pred = pred.to(device)
        pred_len = 1
        while pred_len <= 30:
            tgt_mask = model.generate_square_subsequent_mask(pred.size(dim=0)).type(torch.bool)
            tgt_mask = tgt_mask.to(device)
            hat = model.decode(pred,input_encoder.transpose(0, 1),tgt_mask)
            decode_out = hat.transpose(0, 1)
            prob = model.generator(decode_out[:, -1])
            value, Yt = torch.max(prob, dim=1)
            Yt = Yt.item()
            next_token = torch.tensor([[0]])
            next_token[0,0] = Yt
            next_token = next_token.to(device)
            pred = torch.cat([pred,next_token], dim=0)
            pred_len += 1
            if Yt == vocab['<eos>']:
                break
        
        #detokenize the predicted sentence
        pred_sent = ""
        pred_list = list(pred.cpu().numpy())
        pred_detoken = []
        for m in range(0,len(pred_list)):
            de_tok =[k for k, v in vocab.items() if v == pred_list[m]][0]
            if de_tok  == '<bos>' or de_tok  =='<eos>':
                continue
            else:
                pred_detoken.append(de_tok)
        
        pred_sent = " ".join(pred_detoken)
        
        w+=1

        #detokenize the target sentence
        score = 0
        #iterate all captions
        for l in range(5):
            tgt = test_target[l][i]
            target_sent = ""
            target_detoken = []
            for m in range(0,tgt.size(dim=0)):
                de_tok = [k for k, v in vocab.items() if v == tgt[m].numpy()][0]
                if de_tok  == '<bos>' or de_tok  == '<eos>':
                    continue
                else:
                    target_detoken.append(de_tok)
    
            target_sent = " ".join(target_detoken)
            #print(target_sent)
    
            #Check Bleu score
            score +=  get_bleu(hypotheses=pred_sent.split(), reference=target_sent.split())
        score = score/5
        total_score += score
        print(str(w) + '/500  processed..'+'  score: '+str(score)+"   "+ "total: "+str(total_score)+" "*20,end='\r', flush=True)
        
        
    print("result",total_score/(500))   
    return total_score/(500)

#function to load parameters and test model
def load_test(test_image,test_target,test_model):
    
    #Get model parameters
    test_model.load_state_dict(torch.load('saved_model.pt'))
    test_model.eval()
    total_score = 0
    
    test_data = Dataset(test_image,test_target[0])
    w = 0
    for i in range(len(test_data)):
        src,tgt = test_data[i]
        src = src.to(device)
        logits_, input_encoder = test_model.vit(src.unsqueeze(0))
        input_encoder = input_encoder.to(device)
        pred = torch.tensor([[0]])
        pred[0,0] = vocab['<bos>']
        pred = pred.to(device)
        pred_len = 1
        while pred_len <= 30:
            tgt_mask = test_model.generate_square_subsequent_mask(pred.size(dim=0)).type(torch.bool)
            tgt_mask = tgt_mask.to(device)
            hat = test_model.decode(pred,input_encoder.transpose(0, 1),tgt_mask)
            decode_out = hat.transpose(0, 1)
            prob = test_model.generator(decode_out[:, -1])
            value, Yt = torch.max(prob, dim=1)
            Yt = Yt.item()
            next_token = torch.tensor([[0]])
            next_token[0,0] = Yt
            next_token = next_token.to(device)
            pred = torch.cat([pred,next_token], dim=0)
            pred_len += 1
            if Yt == vocab['<eos>']:
                break
        
        #detokenize the predicted sentence
        pred_sent = ""
        pred_list = list(pred.cpu().numpy())
        pred_detoken = []
        for m in range(0,len(pred_list)):
            de_tok =[k for k, v in vocab.items() if v == pred_list[m]][0]
            if de_tok  == '<bos>' or de_tok  =='<eos>':
                continue
            else:
                pred_detoken.append(de_tok)
        
        pred_sent = " ".join(pred_detoken)
        
        w+=1

        #detokenize the target sentence
        score = 0
        for l in range(5):
            tgt = test_target[l][i]
            target_sent = ""
            target_detoken = []
            for m in range(0,tgt.size(dim=0)):
                de_tok = [k for k, v in vocab.items() if v == tgt[m].numpy()][0]
                if de_tok  == '<bos>' or de_tok  == '<eos>':
                    continue
                else:
                    target_detoken.append(de_tok)
    
            target_sent = " ".join(target_detoken)
            #print(target_sent)
    
            #Check Bleu score
            score +=  get_bleu(hypotheses=pred_sent.split(), reference=target_sent.split())
        score = score/5
        total_score += score
        print(str(w) + '/500  processed..'+'  score: '+str(score)+"   "+ "total: "+str(total_score)+" "*20,end='\r', flush=True)

        
    print("result",total_score/(500))   
    return total_score/(500)

#Function to plot losses over epochs
def print_loss(train_losses,val_losses):
    epochs = np.arange(1, len(train_losses)+ 1)
    plt.plot(epochs, train_losses, label = "train_loss")
    plt.plot(epochs, val_losses,label = "val_loss")
    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.title('Loss over epoch')

    plt.show()
    plt.savefig("loss.png")

#Function to plot bleu score over epochs
def print_bleu(bleu_scores,epoch):
    epochs = np.arange(1, len(bleu_scores)+1)
    plt.plot(epochs, bleu_scores)
    plt.xlabel('epoch')

    plt.ylabel('bleu')

    plt.title('bleu_score')

    plt.show()
    plt.savefig("score.png")

    
    
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    #train(model,train_image,val_image,train_target,val_target,optimizer,test_image,test_target)
    model = model.to('cpu')
    #test(test_image,test_target)
    test_model = MyTransformer(
    NUM_DECODER_LAYERS,
    EMB_SIZE, NHEAD,
    TGT_VOCAB_SIZE,
    FFN_HID_DIM)

    for p in test_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


    test_model = test_model.to(device)
    load_test(test_image,test_target,test_model)
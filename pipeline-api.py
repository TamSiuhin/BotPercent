import tweepy
import json
from transformers import pipeline
from transformers import T5Tokenizer, T5EncoderModel
import user_features as uf
import json
import pickle
from model import MLP, MLP_text
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import argparse
import os
os.environ['http_proxy'] = 'http://127.0.0.1:15236'
os.environ['https_proxy'] = 'http://127.0.0.1:15236' 

parser = argparse.ArgumentParser(description="Bot Percentage API")
parser.add_argument("--username", type=str, default=None)
parser.add_argument("--device", type=str, default='cpu')

args = parser.parse_args()
online=True

with open("api.json") as f:
    api = json.load(f)
    
api = api[0]
api_key = api["key"]
api_secret = api["key_secret"]
access_token = api["access_token"]
access_token_secret = api["access_token_secret"]
bearer_token = api["bearer_token"]

auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, proxy="127.0.0.1:15236", wait_on_rate_limit=True)

with open("./checkpoint/RandomForest.pkl", 'rb') as f:
    RF = pickle.load(f)
    
with open("./checkpoint/Adaboost.pkl", 'rb') as f:
    Adaboost = pickle.load(f)


mlp_hgt = MLP(5, 3, 768, 768, 1024, 0.3)
hgt_state = torch.load('./checkpoint/HGT.pt', map_location=args.device)
mlp_hgt.load_state_dict(hgt_state)

mlp_simplehgn = MLP(5, 3, 768, 768, 1024, 0.3)
simplehgn_state = torch.load('./checkpoint/SHGN.pt', map_location=args.device)
mlp_simplehgn.load_state_dict(simplehgn_state)

mlp_rgt = MLP(5, 3, 768, 768, 1024, 0.3)
RGT_state = torch.load('./checkpoint/RGT.pt', map_location=args.device)
mlp_rgt.load_state_dict(RGT_state)

mlp_rgcn = MLP(5, 3, 768, 768, 1024, 0.3)
rgcn_state = torch.load('./checkpoint/BotRGCN.pt', map_location=args.device)
mlp_rgcn.load_state_dict(rgcn_state)

mlp_roberta = MLP_text(768, 768, 128, 0.3)
roberta_state = torch.load('./checkpoint/text-RoBERTa.pt', map_location=args.device)
mlp_roberta.load_state_dict(roberta_state)

mlp_t5 = MLP_text(512, 512, 128, 0.3)
t5_state = torch.load('./checkpoint/text-T5.pt', map_location=args.device)
mlp_t5.load_state_dict(t5_state)

mlp_hgt.eval()
mlp_simplehgn.eval()
mlp_rgt.eval()
mlp_rgcn.eval()
mlp_t5.eval()
mlp_roberta.eval()

pred_all = []
roberta_extract = pipeline('feature-extraction',
                             model='roberta-base',
                             tokenizer='roberta-base',
                             device=args.device,
                             padding=True, 
                             truncation=True,
                             max_length=50, 
                             add_special_tokens=True)

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5EncoderModel.from_pretrained('t5-small')
t5_extract = pipeline('feature-extraction',
                        model=model,
                        tokenizer=tokenizer,
                        device=args.device,
                        padding=True, 
                        truncation=True,
                        max_length=50, 
                        add_special_tokens = True)

def detect(username):
    user = api.user_timeline(
        screen_name=username,
        tweet_mode='extended',
        count=20
    )

    upper, lower = uf.upper_lower_username_cnt(user, online)
    entropy = uf.ShannonEntropyAndNomalize(user, online)
    # print("2: {}".format(time.time() - start))
    # pos_cnt = uf.POS_feature(user)
    # print("3: {}".format(time.time() - start))


    single_feature = [
        uf.status_cnt(user, online),
        uf.followers_cnt(user, online),
        uf.following_cnt(user, online),
        uf.listed_cnt(user, online),
        uf.default_profile_image(user, online),
        uf.is_verified(user, online),
        uf.get_user_id(user, online),
        uf.has_description(user, online),
        uf.protected(user, online),
        uf.get_digits_screen_name(user, online),
        uf.get_digits_username(user, online),
        uf.has_location(user, online),
        uf.num_of_hashtags(user, online),
        uf.num_of_URLs(user, online),
        uf.has_url(user, online),
        uf.user_age(user, 'Twibot-22', online),
        uf.has_bot_word_in_description(user, online),
        uf.has_bot_word_in_screen_name(user, online),
        uf.has_bot_word_in_username(user, online),
        uf.get_screen_name_length(user, online),
        uf.get_username_length(user, online),
        uf.get_description_length(user, online),
        # uf.get_followees(user),
        # uf.get_followers(user),
        upper,
        lower,
        uf.get_followers_followees(user, online),
        uf.get_number_count_in_screen_name(user, online),
        uf.get_number_count_in_username(user, online),
        uf.hashtags_count_in_username(user, online),
        uf.hashtags_count_in_description(user, online),
        uf.urls_count_in_description(user, online),
        uf.def_image(user, online),
        uf.def_profile(user, online),
        uf.tweet_freq(user, online),
        uf.followers_growth_rate(user, online),
        uf.friends_growth_rate(user, online),           
        uf.screen_name_unicode_group(user, online),
        uf.des_sentiment_score(user, online),
        uf.lev_distance_username_screen_name(user, online),
        entropy
    ]

    single_feature = np.array(single_feature).reshape(1,-1)
    t5_des = uf.get_des_embedding(user, t5_extract, 512)
    t5_tweet = uf.get_tweets_embedding(user, t5_extract, 512)
    roberta_des = uf.get_des_embedding(user, roberta_extract, 768)
    roberta_tweet = uf.get_tweets_embedding(user, roberta_extract, 768)

    num = uf.get_num_feat(user)
    cat = uf.get_cat_feat(user)

    RF_pred = RF.predict_proba(single_feature)
    Ada_pred = Adaboost.predict_proba(single_feature)

    hgt_pred = mlp_hgt(num, cat, roberta_tweet, roberta_des)
    simplehgn_pred = mlp_simplehgn(num, cat, roberta_tweet, roberta_des)
    rgcn_pred = mlp_rgcn(num, cat, roberta_tweet, roberta_des)
    rgt_pred = mlp_rgt(num, cat, roberta_tweet, roberta_des)

    roberta_pred = mlp_roberta(roberta_tweet, roberta_des)
    t5_pred = mlp_t5(t5_tweet, t5_des)

    # calibration
    hgt_pred = torch.softmax(hgt_pred/1.828, dim=-1) # 
    simplehgn_pred = torch.softmax(simplehgn_pred/1.818, dim=-1) # 
    rgt_pred = torch.softmax(rgt_pred/1.826, dim=-1) # 
    rgcn_pred = torch.softmax(rgcn_pred/1.827, dim=-1) # 
    
    roberta_pred = torch.softmax(roberta_pred/1.560, dim=-1) # 
    t5_pred = torch.softmax(t5_pred/1.552, dim=-1)

    RF_pred = torch.softmax(torch.from_numpy(RF_pred) /1.415, dim=-1)
    Ada_pred = torch.softmax(torch.from_numpy(Ada_pred)/1.498, dim=-1)

    pred_stack = torch.stack([
                        RF_pred, 
                        Ada_pred, 
                        hgt_pred.detach().cpu(), 
                        simplehgn_pred.detach().cpu(), 
                        rgt_pred.detach().cpu(),
                        rgcn_pred.detach().cpu(),
                        roberta_pred.detach().cpu(),
                        t5_pred.detach().cpu()
                        ], dim=-1)
        
    weight = torch.tensor([[1.0990, 1.0991, 0.9009, 0.9008, 0.9008, 0.9008, 0.8996, 0.9006]]).t().double()
    pred_all = torch.matmul(pred_stack, weight).squeeze(-1)
    pred_binary = torch.argmax(pred_all, dim=1)
    
    print('-'*100)
    print("pred_score: \t human: {}, bot: {}".format(pred_all[:,0].item(), pred_all[:,1].item()))
    if pred_binary==0:
        print('human!')
    else:
        print("BOT!")
    print('-'*100)

detect(args.username)
import user_features as uf
import json
import pickle
from model import MLP, MLP_text
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import tqdm


with open("./checkpoint/Twibot-22_RandomForest_mix.pkl", 'rb') as f:
    RF = pickle.load(f)
 
with open("./checkpoint/Twibot-22_Adaboost_mix.pkl", 'rb') as f:
    Adaboost = pickle.load(f)   

with open("./checkpoint/Twibot-22_xgboost_mix.pkl", 'rb') as f:
    xgboost = pickle.load(f)
    
# with open("/data3/botp/LOBO/Twibot-22/model.pkl", 'rb') as f:
#     SVC = pickle.load(f)

mlp_hgt = MLP(5, 3, 768, 768, 1024, 0.3)
hgt_state = torch.load('/data3/botp/pipeline_final/ckpt_old/HGT.pt')
mlp_hgt.load_state_dict(hgt_state)

mlp_simplehgn = MLP(5, 3, 768, 768, 1024, 0.3)
simplehgn_state = torch.load('/data3/botp/pipeline_final/ckpt_old/SHGN.pt')
mlp_simplehgn.load_state_dict(simplehgn_state)

mlp_rgt = MLP(5, 3, 768, 768, 1024, 0.3)
RGT_state = torch.load('/data3/botp/pipeline_final/ckpt_old/RGT.pt')
mlp_rgt.load_state_dict(RGT_state)

mlp_RGCN = MLP(5, 3, 768, 768, 1024, 0.3)
RGCN_state = torch.load('/data3/botp/pipeline_final/ckpt_old/BotRGCN.pt')
mlp_RGCN.load_state_dict(RGCN_state)

mlp_roberta = MLP_text(768, 768, 128, 0.3)
roberta_state = torch.load('/data3/botp/pipeline_final/checkpoint/text-RoBERTa.pt')
mlp_roberta.load_state_dict(roberta_state)

mlp_t5 = MLP_text(512, 512, 128, 0.3)
t5_state = torch.load('/data3/botp/pipeline_final/checkpoint/text-T5.pt')
mlp_t5.load_state_dict(t5_state)

mlp_hgt.eval()
mlp_simplehgn.eval()
mlp_rgt.eval()
mlp_RGCN.eval()
mlp_rgt.eval()
mlp_RGCN.eval()
mlp_roberta.eval()
mlp_t5.eval()

des_all = torch.load('/data2/whr/TwiBot22-baselines/src/BotRGCN_old/data_twi22/des_tensor.pt')
tweet_all = torch.load('/data2/whr/TwiBot22-baselines/src/BotRGCN_old/data_twi22/tweets_tensor.pt')
numeric_all = torch.load('/data2/whr/TwiBot22-baselines/src/BotRGCN_old/data_twi22/num_properties_tensor.pt')
cat_all = torch.load('/data2/whr/TwiBot22-baselines/src/BotRGCN_old/data_twi22/cat_properties_tensor.pt')
tweet_all_t5 = torch.load('/data2/whr/TwiBot22-baselines/data/twibot22/T5-tweet/tweets_tensor.pt')
des_all_t5 = torch.load('/data2/whr/TwiBot22-baselines/data/twibot22/T5-tweet/des_tensor.pt')

all_user = json.load(open('/data3/botp/combine_data/combine_graph.json'))
id2idx = json.load(open('/data3/botp/combine_data/id2idx_graph.json'))

# with open("/data3/botp/pipeline/test4.json", 'r') as f:
for i in range(10):
    with open('/data2/whr/TwiBot22-baselines/datasets/Twibot-22/domain/user{}.json'.format(i), 'r') as f:
        
        
        pred_all = []
        idx_list = []
        data = json.load(f)
        for idx, user in tqdm.tqdm(enumerate(data)):
            # user = all_user[id2idx["Twibot-22:"+user]]
            idx_list.append(id2idx["Twibot-22:"+user])
            
            # upper, lower = uf.upper_lower_username_cnt(user)
            # # print("1: {}".format(time.time() - start))
            # entropy = uf.ShannonEntropyAndNomalize(user)
            # # print("2: {}".format(time.time() - start))
            # # pos_cnt = uf.POS_feature(user)
            # # print("3: {}".format(time.time() - start))
            # single_feature = [
            #     uf.status_cnt(user),
            #     uf.followers_cnt(user),
            #     uf.following_cnt(user),
            #     uf.listed_cnt(user),
            #     uf.default_profile_image(user),
            #     uf.is_verified(user),
            #     uf.get_user_id(user),
            #     uf.has_description(user),
            #     uf.protected(user),
            #     uf.get_digits_screen_name(user),
            #     uf.get_digits_username(user),
            #     uf.has_location(user),
            #     uf.num_of_hashtags(user),
            #     uf.num_of_URLs(user),
            #     uf.has_url(user),
            #     uf.user_age(user),
            #     uf.has_bot_word_in_description(user),
            #     uf.has_bot_word_in_screen_name(user),
            #     uf.has_bot_word_in_username(user),
            #     uf.get_screen_name_length(user),
            #     uf.get_username_length(user),
            #     uf.get_description_length(user),
            #     # uf.get_followees(user),
            #     # uf.get_followers(user),
            #     upper,
            #     lower,
            #     uf.get_followers_followees(user),
            #     uf.get_number_count_in_screen_name(user),
            #     uf.get_number_count_in_username(user),
            #     uf.hashtags_count_in_username(user),
            #     uf.hashtags_count_in_description(user),
            #     uf.urls_count_in_description(user),
            #     uf.def_image(user),
            #     uf.def_profile(user),
            #     uf.tweet_freq(user),
            #     uf.followers_growth_rate(user),
            #     uf.friends_growth_rate(user),           
            #     uf.screen_name_unicode_group(user),
            #     uf.des_sentiment_score(user),
            #     uf.lev_distance_username_screen_name(user),
            #     entropy
            # ]
            
            # single_feature = np.array(single_feature).reshape(1,-1)
            
    idx_all = np.array(idx_list)
    des = des_all[idx_all]
    tweet = tweet_all[idx_all]
    cat = cat_all[idx_all]
    num = numeric_all[idx_all]
    tweet_t5 = tweet_all_t5[idx_all]
    des_t5 = des_all_t5[idx_all]

    features = np.load('/data3/botp/combine_data/feature_mix_fix.npy')
    features = features[idx_all]

    hgt_pred = mlp_hgt(num, cat, tweet, des)
    simplehgn_pred = mlp_simplehgn(num, cat, tweet, des)
    rgt_pred = mlp_rgt(num, cat, tweet, des)
    rgcn_pred = mlp_RGCN(num, cat, tweet, des)

    roberta_pred = mlp_roberta(tweet, des)
    t5_pred = mlp_t5(tweet_t5, des_t5)
    
    RF_pred = RF.predict_proba(features)
    Ada_pred = Adaboost.predict_proba(features)
    xg_pred = xgboost.predict_proba(features)

    hgt_pred = torch.softmax(hgt_pred/1.828, dim=-1) # 
    simplehgn_pred = torch.softmax(simplehgn_pred/1.818, dim=-1) # 
    rgt_pred = torch.softmax(rgt_pred/1.826, dim=-1) # 
    rgcn_pred = torch.softmax(rgcn_pred/1.827, dim=-1) # 
    roberta_pred = torch.softmax(roberta_pred/1.560, dim=-1) # 
    t5_pred = torch.softmax(t5_pred/1.552, dim=-1) # 
    
    RF_pred = torch.softmax(torch.from_numpy(RF_pred) /1.415, dim=-1)
    Ada_pred = torch.softmax(torch.from_numpy(Ada_pred)/1.498, dim=-1)
    # xg_pred = torch.from_numpy(xg_pred) #/1.145, dim=1)
    # print("RF: {}".format(RF_pred))
    # print("Ada: {}".format(Ada_pred))

    # print(RF_pred)
    # print(Ada_pred)
    # SVC_pred = SVC.predict_proba(single_feature)
    # print(hgt_pred.shape)
    # print(RF_pred.shape)
    # pred_all =  

    combine = torch.stack([
                        RF_pred, 
                        Ada_pred, 
                        hgt_pred.detach().cpu(), 
                        simplehgn_pred.detach().cpu(), 
                        rgt_pred.detach().cpu(),
                        rgcn_pred.detach().cpu(),
                        # roberta_pred.detach().cpu(),
                        # t5_pred.detach().cpu()
                        ], dim=-1)
    
    pred_all.append(combine)
    pred_all = torch.cat(pred_all, dim=0)
    pred_all = pred_all.mean(dim=-1)

    # pred_all = 0.2*RF_pred + 0.2*Ada_pred + 0.4*hgt_pred.detach().cpu()+0.3*simplehgn_pred.detach().cpu()
    # + 0.1*rgt_pred.detach().cpu() + 0.1*rgcn_pred.detach().cpu()+ 0.2 *roberta_pred.detach().cpu()+0.2*t5_pred.detach().cpu()
    # # pred_all = torch.softmax(pred_all, dim=1)
    
    pred_binary = torch.zeros(pred_all.shape[0])
    pred_binary[pred_all[:,1]>=0.55] = 1

    print("threshold 0.55: {}".format((pred_binary==1).sum()))
    
    pred_binary = torch.zeros(pred_all.shape[0])
    pred_binary[pred_all[:,1]>=0.6] = 1

    print("threshold 0.6: {}".format((pred_binary==1).sum()))
    
    pred_binary = torch.zeros(pred_all.shape[0])
    pred_binary[pred_all[:,1]>=0.65] = 1

    print("threshold 0.65: {}".format((pred_binary==1).sum()))
    
    pred_binary = torch.zeros(pred_all.shape[0])
    pred_binary[pred_all[:,1]>=0.7] = 1

    print("threshold 0.7: {}".format((pred_binary==1).sum()))
    
    pred_binary = torch.zeros(pred_all.shape[0])
    pred_binary[pred_all[:,1]>=0.75] = 1

    print("threshold 0.75: {}".format((pred_binary==1).sum()))
    
    pred_binary = torch.zeros(pred_all.shape[0])
    pred_binary[pred_all[:,1]>=0.8] = 1

    print("threshold 0.8: {}".format((pred_binary==1).sum()))
    
    # pred_all[pred_all[:,1]>=0.6]
    pred_binary = torch.argmax(pred_all, dim=1)
    
    print("0.5: {}".format((pred_binary==1).sum()))

    # all_label = torch.load("/data3/botp/combine_data/emb/label.pt")
    # test_idx = torch.load("/data3/botp/combine_data/emb/test_idx.pt")
    # test_label = all_label[test_idx]
    
    # # test_label = np.load('/data3/botp/combine_data/label_test2.npy')
    
    # # print(test_label.shape)
    # # print(pred_binary.shape)
    # acc = accuracy_score(test_label, pred_binary)
    # f1 = f1_score(test_label, pred_binary.cpu())
    # precision = precision_score(test_label, pred_binary.cpu())
    # recall = recall_score(test_label, pred_binary.cpu())
    
    # print("acc: {}".format(acc))
    # print("f1: {}".format(f1))
    # print("precision: {}".format(precision))
    # print("recall: {}".format(recall))
    
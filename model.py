import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, numeric_num, cat_num, tweet_dim, des_dim, hidden_size=1024, dropout=0.3):
        super(MLP, self).__init__()
        self.linear_numeric = nn.Linear(numeric_num, int(hidden_size//4))
        self.linear_cat = nn.Linear(cat_num, int(hidden_size//4))
        self.linear_tweet = nn.Linear(tweet_dim, int(hidden_size//4))
        self.linear_des = nn.Linear(des_dim, int(hidden_size//4))
        
        self.lin1 = torch.nn.Linear(hidden_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, 2)
        self.act = torch.nn.ReLU()
        self.drop_ratio = dropout
        
    def forward(self, numeric, cat, tweet, des):
        numeric = F.dropout(F.relu(self.linear_numeric(numeric)), p=self.drop_ratio, training=self.training)
        cat = F.dropout(F.relu(self.linear_cat(cat)), p=self.drop_ratio, training=self.training)
        tweet = F.dropout(F.relu(self.linear_tweet(tweet)), p=self.drop_ratio, training=self.training)
        des = F.dropout(F.relu(self.linear_des(des)), p=self.drop_ratio, training=self.training)
        
        feature = torch.cat([numeric, cat, tweet, des], dim=-1)
        feature = self.lin1(feature)
        feature = F.relu(feature)
        feature = F.dropout(feature, p=self.drop_ratio, training=self.training)
        pred = self.lin2(feature)
        
        return pred
    
# class MLP_text(torch.nn.Module):
#     def __init__(self, tweet_dim, des_dim, hidden_size=1024, dropout=0.3):
#         super(MLP_text, self).__init__()
#         self.dropout = dropout
#         self.lin_twe = torch.nn.Linear(tweet_dim, hidden_size//2)
#         self.lin_des = torch.nn.Linear(des_dim, hidden_size//2)
#         self.lin1 = torch.nn.Linear(hidden_size, hidden_size)
#         self.lin2 = torch.nn.Linear(hidden_size, 2)
        
#     def forward(self, tweet, des):
#         tweet = F.dropout(F.relu(self.lin_twe(tweet)), p=self.dropout, training=self.training)
#         des = F.dropout(F.relu(self.lin_des(des)), p=self.dropout, training=self.training)
        
#         feature = torch.cat([tweet, des], dim=-1)
#         feature = self.lin1(feature)
#         feature = F.relu(feature)
#         feature = F.dropout(feature, p=self.dropout, training=self.training)
#         pred = self.lin2(feature)
        
#         return pred
    
class MLP_text(torch.nn.Module):
    def __init__(self, tweet_dim, des_dim, hidden_size=1024, dropout=0.3):
        super(MLP_text, self).__init__()
        self.dropout = dropout
        self.lin_twe = torch.nn.Linear(tweet_dim, hidden_size//2)
        self.lin_des = torch.nn.Linear(des_dim, hidden_size//2)
        self.lin1 = torch.nn.Linear(hidden_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, 2)
        
    def forward(self, tweet, des):
        tweet = F.dropout(F.relu(self.lin_twe(tweet)), p=self.dropout, training=self.training)
        des = F.dropout(F.relu(self.lin_des(des)), p=self.dropout, training=self.training)
        
        feature = torch.cat([tweet, des], dim=-1)
        feature = self.lin1(feature)
        feature = F.relu(feature)
        feature = F.dropout(feature, p=self.dropout, training=self.training)
        pred = self.lin2(feature)
        
        return pred
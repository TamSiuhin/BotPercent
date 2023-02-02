import re
import torch

def status_cnt(user, online=False, single=False):
    if not online:
        return user['public_metrics']['tweet_count']
    elif single:
        return user._json['statuses_count']
    else:
        return user._json['user']['statuses_count']
    
def followers_cnt(user, online=False, single=False):
    if not online:
        return user['public_metrics']['followers_count']
    if single:
        return user._json['followers_count']
    else:
        return user._json['user']['followers_count']
    
def following_cnt(user, online=False, single=False):
    if not online:
        return user['public_metrics']['following_count']
    elif single:
        return user._json['friends_count']
    else:
        return user._json['user']['friends_count']
    
def listed_cnt(user, online=False, single=False):
    if not online:
        return user['public_metrics']['listed_count']
    elif single:
        return user._json['listed_count']
    else:
        return user._json['user']['listed_count']
    
def default_profile_image(user, online=False, single=False):
    if not online:
        result = int(user['profile_image_url'].find('default_profile_normal') == -1)
        return result
    elif single:
        result = int(user._json['profile_image_url'].find('default_profile_normal') == -1)
        return result
    else:
        result = int(user._json['user']['profile_image_url'].find('default_profile_normal') == -1)
        return result
    
def is_verified(user, online=False, single=False):
    if not online:
        return int(user['verified'])
    elif single:
        return int(user._json['verified'])
    else:
        return int(user._json['user']['verified'])
    
def get_user_id(user, online=False, single=False):
    if not online:
        return int(user['id'].replace('u',''))
    elif single:
        return int(user._json['id'])
    else:
        return int(user._json['user']['id'])
    
def get_username(user, online=False, single=False):
    if not online:
        return user['username']
    elif single:
        return user._json['screen_name']
    else:
        return user._json['user']['screen_name']
    
def get_screen_name(user, online=False, single=False):
    if not online:    
        return user['name']
    elif single:
        return user._json['name']
    else:
        return user._json['user']['name']
    
def has_description(user, online=False, single=False):
    if not online:
        des = user['description']
    elif single:
        des = user._json['description']
    else:
        des = user._json['user']['description']
        
    if des is None:
        return 0
    else:
        return 1

def protected(user, online=False, single=False):
    if not online:
        return int(user['protected'])
    elif single:
        return int(user._json['protected'])
    else:
        return int(user._json['user']['protected'])
    
# import json
# def screen_name_likelihood(user):
#     bi_gram_likelihood = json.load(open('tmp/{}/bi_gram_likelihood.json'.format(dataset)))
#     value = value.strip()
#     ans = 1
#     for index in range(len(value) - 1):
#         bi_gram = value[index] + value[index + 1]
#         ans *= bi_gram_likelihood[bi_gram] ** (1.0 / (len(value) - 1))
#     return ans

def get_digits_screen_name(user, online=False, single=False):
    if not online:
        name = user['name']
    elif single:
        name = user._json['name']
    else:
        name = user._json['user']['name']
        
    cnt = 0
    for i in name:
        if '0'<=i and i<='9':
            cnt += 1
    return cnt

def get_digits_username(user, online=False, single=False):
    if not online:
        name = user["username"]
    elif single:
        name = user._json['screen_name']
    else:
        name = user._json['user']['screen_name']

    cnt = 0
    for i in name:
        if '0'<=i and i<='9':
            cnt+=1
    return cnt
    
def has_location(user, online=False, single=False):
    if not online:
        location = user['location']
    elif single:
        location = user._json['location']
    else:
        location = user._json['user']['location']
        
    if location is None:
        return 0
    else:
        return 1

def num_of_hashtags(user, online=False, single=False):
    
    try:
        if not online:  
            return len(user['entities']['description']['hashtags'])
        else:
            return len(user._json['entities']['description']['hashtags'])
    except:
        return 0
    
def num_of_URLs(user, online=False, single=False):
    try:
        if not online:
            return len(user['entities']['url']['urls'])
        elif single:
            return len(user._json['entities']['description']['urls'])
        else:
            return len(user._json['user']['entities']['description']['urls'])
        
    except:
        return 0
    
def has_url(user, online=False, single=False):
    if not online:
        url = user['url']
    elif single:
        url = user._json['url']
    else:
        url = user._json['user']['url']
        
    if url is None:
        return 0
    else:
        return 1

from datetime import datetime
def user_age(user, dataset="Twibot-22", online=False, single=False):
    if not online:
        created_at = user['created_at']
    elif single:
        created_at = user._json['created_at']
    else:
        created_at = user._json['user']['created_at']
        
    if created_at is None:
        return 365 * 2
    created_at = created_at.strip()
    if dataset in ['Twibot-20', 'gilani-2017', 'cresci-stock-2018', 'cresci-rtbust-2019',
                   'cresci-2017', 'cresci-2015', 'botometer-feedback-2019']:
        mode = '%a %b %d %H:%M:%S %z %Y'
    elif dataset in ['Twibot-22']:
        mode = '%Y-%m-%d %H:%M:%S%z'
    elif dataset in ['midterm-2018']:
        mode = '%a %b %d %H:%M:%S %Y'
    else:
        raise KeyError
    
    collect_time = datetime.strptime('20{} Dec 31'.format(dataset[-2:]), '%Y %b %d')
    
    if online:
        mode = '%a %b %d %H:%M:%S %z %Y'
        collect_time = datetime.strptime('Tue Sep 5 00:00:00 +0000 2020','%a %b %d %X %z %Y')
        
    if created_at.find('L') != -1:
        created_time = datetime.fromtimestamp(int(created_at.replace('000L', '')))
    else:
        created_time = datetime.strptime(created_at, mode)
    
    created_time = created_time.replace(tzinfo=None)
    collect_time = collect_time.replace(tzinfo=None)
    difference = collect_time - created_time
    return difference.days

def has_bot_word_in_description(user, online=False, single=False):
    
    if has_description(user, online, single):
        if not online:
            matchObj = re.search('bot', user['description'], flags=re.IGNORECASE)
        elif single:
            matchObj = re.search('bot', user._json['description'], flags=re.IGNORECASE)
        else:
            matchObj = re.search('bot', user._json['user']['description'], flags=re.IGNORECASE)
        if matchObj:
            return 1
        else:
            return 0
    else:
        return 0


def has_bot_word_in_screen_name(user, online=False, single=False):
    screen_name = get_screen_name(user, online, single)
    
    if screen_name is None:
        return 0
    else:
        matchObj = re.search('bot', screen_name, flags=re.IGNORECASE)
        if matchObj:
            return 1
        else:
            return 0


def has_bot_word_in_username(user, online=False, single=False):
    
    username = get_username(user, online, single)
    if username is None:
        return False
    else:
        matchObj = re.search('bot', username, flags=re.IGNORECASE)
        if matchObj:
            return 1
        else:
            return 0


def get_screen_name_length(user, online=False, single=False):
    screen_name = get_screen_name(user, online, single)
    
    if screen_name is None:
        return 0
    else:
        return len(screen_name)


def get_username_length(user, online=False, single=False):
    username = get_username(user, online, single)
    if username is None:
        return 0
    else:
        return len(username)


def get_description_length(user, online=False, single=False):
    if has_description(user, online, single):
        if not online:
            return len(user['description'])
        elif single:
            return len(user._json['description'])
        else:
            return len(user._json['user']['description'])
    else:
        return 0


def get_followees(user, online=False, single=False):
    if not online:
        following_count = user['public_metrics']['following_count']
    elif single:
        following_count = user._json['friends_count']
    else:
        following_count = user._json['user']['friends_count']
    
    if following_count is None:
        return 0
    else:
        return int(following_count)


def get_followers(user, online=False, single=False):
    if not online:
        followers_count = user['public_metrics']['followers_count']
    elif single:
        followers_count = user._json['followers_count']
    else:
        followers_count = user._json['user']['followers_count']
        
    if followers_count is None:
        return 0
    else:
        return int(followers_count)

def upper_lower_username_cnt(user, online=False, single=False):
    if not online:
        str1 = user['username']
    elif single:
        str1 = user._json['screen_name']
    else:
        str1 = user._json['user']['screen_name']
        
    big_num = 0  
    small_num = 0   

    for c in str1:
        if c.isupper():
            big_num += 1
        elif c.islower():
            small_num += 1
            
    return big_num, small_num

def upper_lower_screen_name_cnt(user, online=False, single=False):
    if not online:
        str1 = user['name']
    elif single:
        str1 = user._json['name']
    else:
        str1 = user._json['user']['name']
        
    big_num = 0  
    small_num = 0   

    for c in str1:
        if c.isupper():
            big_num += 1
        elif c.islower():
            small_num += 1
            
    return big_num, small_num


def get_followers_followees(user, online=False, single=False):
    if not online:
        followers_count = user['public_metrics']['followers_count']
        following_count = user['public_metrics']['following_count']
    elif single:
        followers_count = user._json['followers_count']
        following_count = user._json['friends_count']
    else:
        followers_count = user._json['user']['followers_count']
        following_count = user._json['user']['friends_count']
   
    if following_count is None or following_count == 0:
        return 0.0
    else:
        return int(followers_count) / int(following_count)


# maybe r'\d'
def get_number_count_in_screen_name(user, online=False, single=False):
    screen_name = get_screen_name(user, online, single)
    if screen_name is None:
        return 0
    else:
        numbers = re.findall(r'\d+', screen_name)
        return len(numbers)


def get_number_count_in_username(user, online=False, single=False):
    username = get_username(user, online, single)
    if username is None:
        return 0
    else:
        numbers = re.findall(r'\d+', username)
        return len(numbers)


def hashtags_count_in_username(user, online=False, single=False):
    if not online:
        username = user["username"]
    elif single:
        username = user._json['screen_name']
    else:
        username = user._json['user']['screen_name']
        
    if username is None:
        return 0
    else:
        hashtags = re.findall(r'#\w', username)
    return len(hashtags)


def hashtags_count_in_description(user, online=False, single=False):
    if has_description(user, online, single):
        if not online:
            hashtags = re.findall(r'#\w', user['description'])
        elif single:
            hashtags = re.findall(r'#\w', user._json['description'])
        else:
            hashtags = re.findall(r'#\w', user._json['user']['description'])

        return len(hashtags)
    else:
        return 0


def urls_count_in_description(user, online=False, single=False):
    if has_description(user, online, single):
        if not online:
            urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user["description"])
        elif single:
            urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user._json["description"])
        else:
            urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user._json['user']["description"])
        return len(urls)
    else:
        return 0

def def_image(user, online=False, single=False):
    if not online:
        img = user['profile_image_url']
    elif single:
        img = user._json['profile_image_url']
    else:
        img = user._json['user']['profile_image_url']
        
    if img is None:
        return 1
    else:
        return 0

def def_profile(user, online=False, single=False):
    if not online:
        des = user['description']
        loca = user['location']
        url = user['url']
    elif single:
        des = user._json['description']
        loca = user._json['location']
        url = user._json['url']
    else:
        des = user._json['user']['description']
        loca = user._json['user']['location']
        url = user._json['user']['url']
        
    if des is None and loca is None and url is None:
        return 1
    else:
        return 0
    
def tweet_freq(user, online=False, single=False):
    try:
        if not online:
            freq = float(user['pubilic_metrics']['tweet_count'])/user_age(user, online, single)
            return freq
        elif single:
            freq = float(user._json['statuses_count'])/user_age(user, online, single)
        else:
            freq = float(user._json['user']['statuses_count'])/user_age(user, online, single)
    except:
        return 0
    
def followers_growth_rate(user, online=False, single=False):
    try:
        if not online:
            rate = float(user['pubilic_metrics']['followers_count'])/user_age(user, online, single)
            return rate
        elif single:
            rate = float(user._json['followers_count'])/user_age(user, online, single)
            return rate
        else:
            rate = float(user._json['user']['followers_count'])/user_age(user, online, single)
            return rate
    except:
        return 0

def friends_growth_rate(user, online=False, single=False):
    try:
        if not online:
            rate = float(user['pubilic_metrics']['following_count'])/user_age(user, online, single)
            return rate
        elif single:
            rate = float(user._json['friends_count'])/user_age(user, online, single)
            return rate
        else:
            rate = float(user._json['user']['friends_count'])/user_age(user, online, single)
            return rate
    except:
        return 0

from math import log
import numpy as np
def ShannonEntropyAndNomalize(user, online=False, single=False):
    if not online:
        name = user['name']
    elif single:
        name = user._json['name']
    else:
        name = user._json['user']['name']
        
    entropy = []
    for s in name:
        word = {}
        for c in s:
            currentlabel = c
            if c not in word.keys():
                word[c] = 0
            word[currentlabel] += 1
        ShannonEnt = 0.0
        for i in word:
            prob = float(word[i])/len(s)
            ShannonEnt -= prob * log(prob, 2)
        entropy.append(ShannonEnt)
    entropy = np.mean(entropy)
    return entropy

import numpy as np
def Lev_distance(A,B):
    #A = "fafasa"
    #B = "faftreassa"
    
    dp = np.array(np.arange(len(B)+1))
    for i in range(1, len(A)+1):
        temp1 = dp[0]
        dp[0] += 1
        for j in range(1, len(B)+1):
            temp2 = dp[j]
            if A[i-1] == B[j-1]:
                dp[j] = temp1
            else:
                dp[j] = min(temp1, min(dp[j-1], dp[j]))+1
            temp1 = temp2

    return dp[len(B)]


def lev_distance_username_screen_name(user, online=False, single=False):
    if not online:
        name = user['name']
        screen_name = user['username']
    elif single:
        name = user._json['name']
        screen_name = user._json['screen_name']
    else:
        name = user._json['user']['name']
        screen_name = user._json['user']['screen_name']
    dis = Lev_distance(name, screen_name)
    return dis

def screen_name_unicode_group(user, online=False, single=False):
    if not online:
        word = user['name']
    elif single:
        word = user._json['name']
    else:
        word = user._json['user']['name']
        
    classes=[]
    uni_class=np.load('uni_class.npy')
    for i in word:
        uni=0
        for j,k in enumerate(uni_class):
            if(k>ord(i)):
                uni=j-1
                break
        classes.append(uni)
    try:
        return max(classes)
    except:
        return 0

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def des_sentiment_score(user, online=False, single=False):
    if not online:
        text = user['description']
    elif single:
        text = user._json['description']
    else:
        text = user._json['user']['description']
    score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    return score

import spacy
def POS_feature(user, online=False, single=False):
    nlp = spacy.load('en_core_web_trf') 
    if not online:
        text = user['description']
    elif single:
        text = user._json['description']
    else:
        text = user._json['user']['description']
        
    doc = nlp(text)
    
    pos = [token.pos_ for token in doc]
    punct, noun, pron, verb, adv, adj, adp, cconj, num, intj = 0,0,0,0,0,0,0,0,0,0
    
    for p in pos:
        if p=='PUNCT':
            punct += 1
        elif p=='NOUN':
            noun += 1
        elif p=='PRON':
            pron += 1
        elif p=='VERB':
            verb += 1
        elif p=='ADV':
            adv += 1
        elif p=='ADJ':
            adj += 1
        elif p=='ADP':
            adp += 1
        elif p=='CCONJ' or 'SCONJ':
            cconj += 1
        elif p=='NUM':
            num += 1
        elif p=='INTJ':
            intj += 1
    pos_cnt = [punct, noun, pron, verb, adv, adj, adp, cconj, num, intj]
    return pos_cnt

from transformers import pipeline
def get_des_embedding(user, pipeline, dim=768, single=False):
    if not single:
        description = user._json['user']['description']
    else:
        description = user._json['description']
    # feature_extract = pipeline('feature-extraction',
    #                          model='roberta-base',
    #                          tokenizer='roberta-base',
    #                          device=3,
    #                          padding=True, 
    #                          truncation=True,
    #                          max_length=50, 
    #                          add_special_tokens=True)
    rep = torch.zeros(1, dim)
    
    feature = pipeline(description)
    # print(torch.tensor(feature).size())

    for i in feature[0]:
        # print(torch.tensor(i).size())
        rep += torch.tensor(i).unsqueeze(0)
        
    rep = rep/len(feature[0])
    return rep


def get_tweets_embedding(timeline, pipeline, dim=768):
    # feature_extract = pipeline('feature-extraction',
    #                          model='roberta-base',
    #                          tokenizer='roberta-base',
    #                          device=3,
    #                          padding=True, 
    #                          truncation=True,
    #                          max_length=50, 
    #                          add_special_tokens=True)
    rep = torch.zeros(1, dim)
    rep_tmp = torch.zeros(1, dim)
    
    if len(timeline) > 0:
        for i in timeline:
            tweet = i._json['full_text']
            tweet_feat = pipeline(tweet)
            for j in tweet_feat[0]:
                rep_tmp += torch.tensor(j).unsqueeze(0)
            rep_tmp = rep_tmp/len(tweet_feat[0])
            rep += rep_tmp 

        rep = rep / len(timeline)
    return rep

def get_cat_feat(user, single=False):
    if single:
        protect = user._json['protected']
        verified = user._json['verified']
    else:
        protect = user._json['user']['protected']
        verified = user._json['user']['verified']
    default_profile_img = default_profile_image(user, True, single)
    return torch.tensor([[protect, verified, default_profile_img]]).float()

def get_num_feat(user, single=False):
    followers_count = followers_cnt(user, True, single)
    following_count = following_cnt(user, True, single)
    status_count = status_cnt(user, True, single)
    screenname_len = get_screen_name_length(user, True, single)
    active_days = user_age(user, online=True, single=single)
    
    followers_mean = 41230.145237
    followers_std = 602077.3357674746
    follower_count = (followers_count -followers_mean) / followers_std
    
    following_mean = 2250.534189
    following_std = 15782.370203645429
    following_count = (following_count - following_mean) / following_std
    
    status_mean = 10844.5506605
    status_std = 85716.10562431585
    status_count = (status_count - status_mean) / status_std
    
    screenname_mean = 11.157433
    screenname_std = 2.6744636566068722
    screenname_len = (screenname_len - screenname_mean) / screenname_std
    
    active_mean = 2128.597769
    active_std = 1633.2178995661363
    active_days = (active_days - active_mean) / active_std
    
    return torch.tensor([[followers_count,active_days,screenname_len,following_count,status_count]]).float()
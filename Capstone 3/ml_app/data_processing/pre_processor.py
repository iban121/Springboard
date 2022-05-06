#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np

import re
import unidecode
from nltk.stem import PorterStemmer, WordNetLemmatizer
import emoji
import contractions
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[36]:


colloquial_contractions = {'AFAIK': 'As Far As I Know',
    'AFK': 'Away From Keyboard',
    'ASAP': 'As Soon As Possible',
    'ATK': 'At The Keyboard',
    'ATM': 'At The Moment',
    'A3': 'Anytime, Anywhere, Anyplace',
    'BAK': 'Back At Keyboard',
    'BBL': 'Be Back Later',
    'BBS': 'Be Back Soon',
    'BFN': 'Bye For Now',
    'B4N': 'Bye For Now',
    'BRB': 'Be Right Back',
    'BRT': 'Be Right There',
    'BTW': 'By The Way',
    'B4': 'Before',
    'B4N': 'Bye For Now',
    'CU': 'See You',
    'CUL8R': 'See You Later',
    'CYA': 'See You',
    'FAQ': 'Frequently Asked Questions',
    'FC': 'Fingers Crossed',
    'FWIW': 'For What It\'s Worth',
    'FYI': 'For Your Information',
    'GAL': 'Get A Life',
    'GG': 'Good Game',
    'GN': 'Good Night',
    'GMTA': 'Great Minds Think Alike',
    'GR8': 'Great!',
    'G9': 'Genius',
    'IC': 'I See',
    'ICQ': 'I Seek you',
    'ILU': 'I Love You',
    'IMHO': 'In My Humble Opinion',
    'IMO': 'In My Opinion',
    'IOW': 'In Other Words',
    'IRL': 'In Real Life',
    'KISS': 'Keep It Simple, Stupid',
    'LDR': 'Long Distance Relationship',
    'LMAO': 'Laugh My Ass Off',
    'LOL': 'Laughing Out Loud',
    'LTNS': 'Long Time No See',
    'L8R': 'Later',
    'MTE': 'My Thoughts Exactly',
    'M8': 'Mate',
    'NRN': 'No Reply Necessary',
    'OIC': 'Oh I See',
    'OMG': 'Oh My God',
    'PITA': 'Pain In The Ass',
    'PRT': 'Party',
    'PRW': 'Parents Are Watching',
    'QPSA?': 'Que Pasa?',
    'ROFL': 'Rolling On The Floor Laughing',
    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',
    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',
    'SK8': 'Skate',
    'STATS': 'Your sex and age',
    'ASL': 'Age, Sex, Location',
    'THX': 'Thank You',
    'TTFN': 'Ta-Ta For Now!',
    'TTYL': 'Talk To You Later',
    'U': 'You',
    'U2': 'You Too',
    'U4E': 'Yours For Ever',
    'WB': 'Welcome Back',
    'WTF': 'What The Fuck',
    'WTG': 'Way To Go!',
    'WUF': 'Where Are You From?',
    'W8': 'Wait',
    '7K': 'Sick:-D Laugher'}
def expand_colloquialisms(tweet):
    if tweet.upper() in colloquial_contractions.keys():
        return colloquial_contractions[tweet.upper()]
    else:
        return tweet
def expand_contractions(tweet):
    return contractions.fix(tweet)
def demojize(tweet):
    emojis = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags = re.UNICODE)
    cleaned_text = emojis.sub(r'', tweet)
    return cleaned_text
def emojis(tweet):
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(':','')
    cleaned_text = ' '.join(tweet.split())
    return cleaned_text
def remove_html_text(tweet):
    html_text = re.compile(r'<.*?>|\n|\t|\&\w*;z|\$\w*')
    cleaned_text = html_text.sub(r'',tweet)
    return cleaned_text
def remove_url_text(tweet):
    url = re.compile(r'https?://\S+|www\.\S+|http://t.co/|http|https?:\/\/.*\/\w*')
    cleaned_text = url.sub(r'',tweet)
    return cleaned_text
stop_words = ['abyhrgsss', 'ableg', 'abuses', 'absence', 'accidentally', 'accepte', 'acc', 'abcchicago', 'abstract', 'academia', 'abe', 'accordingly', 'abomination', 'abandon', 'accept', 'abject', 'ability', 'abha', 'about', 'accidentalprophecy', 'abstorm', 'acaciapenn', 'abortions', 'able', 'aattamnmd', 'abandoning', 'accidents', 'abusing', 'aaaaaaallll', 'above', 'accepts', 'abc', 'according', 'aace', 'aashiqui', 'aberdeen', 'aamir', 'abceyewitness', 'accidently', 'abran', 'abq', 'aboard', 'abu', 'absurd', 'abs', 'aal', 'abjabhqhe', 'abandoned', 'aberdeenfanpage', 'abolxmhvy', 'accompanying', 'abcnorio', 'absolutely', 'absolute', 'acarewornheart', 'account', 'aar', 'aberystwyth', 'aaaa', 'abgfglhx', 'abysmaljoiner', 'acbryenuo', 'abia', 'abbswinston', 'aan', 'ablaze', 'abgctvfua', 'abortion', 'abuse', 'accountable', 'abnrcr', 'abomb', 'abnzqwlig', 'aawzxykles', 'aaronthefm', 'access', 'abbott', 'aauizggcq', 'abbyairshow', 'aberdeenfc', 'abandonedpics', 'abcnews', 'abninfvet', 'accidentsua', 'absurdly', 'abnlseqb', 'abouts', 'aba', 'aannnnd', 'aaarrrgghhh', 'accionempresa', 'abused', 'ablzmgzv', 'aadzvsr', 'accident', 'absolut', 'aaemiddle', 'abbandoned', 'aaaaaand', 'abbruchsimulator','10', '11', '12', '13', '14', '15', '16', '163', '17', '18', '19',
       '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
       '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
       '42', '43', '44', '45', '46', '47', '48', '49', '51', '52', '54',
       '55', '56', '57', '58', '60', '61', '62', '65', '66', '71', '74',
       '76', '77', '78', '79', '8', '81', '85', '86', '9', '93', '94',
       '95', '96', '97', '98', 'actually', 'after', 'again', 'against',
       'ago', 'air', 'all', 'almost', 'already', 'also', 'always',
       'ambulance', 'america', 'amp', 'and', 'angry', 'annihilated',
       'annihilation', 'another', 'any', 'anyone', 'anything',
       'apocalypse', 'are', 'area', 'army', 'around', 'arson', 'arsonist',
       'ass', 'attack', 'attacked', 'august', 'australia', 'away', 'baby',
       'back', 'bad', 'battle', 'beautiful', 'because', 'become', 'been',
       'before', 'behind', 'being', 'believe', 'best', 'big', 'bioterror',
       'bioterrorism', 'black', 'blast', 'blood', 'bloody', 'boat',
       'body', 'bomb', 'bombed', 'both', 'boy', 'breaking', 'brown',
       'building', 'buildings', 'burned', 'burning', 'bus', 'bush',
       'business', 'but', 'call', 'came', 'can', 'cannot', 'car', 'care',
       'casualty', 'catastrophe', 'catastrophic', 'center', 'centre',
       'change', 'check', 'chemical', 'chicago', 'child', 'children',
       'china', 'city', 'cliff', 'climate', 'collapse', 'collapsed',
       'collided', 'collision', 'come', 'coming', 'content', 'control',
       'cool', 'could', 'country', 'couple', 'crash', 'crashed', 'crazy',
       'cross', 'cyclone', 'daily', 'damage', 'damn', 'danger', 'day',
       'days', 'dead', 'deal', 'death', 'deaths', 'deluged', 'demolished',
       'demolition', 'derail', 'derailed', 'destroy', 'destroyed',
       'destruction', 'detonate', 'detonation', 'devastated',
       'devastation', 'did', 'die', 'disaster', 'displaced', 'does',
       'doing', 'done', 'down', 'drive', 'drowned', 'drowning', 'during',
       'dust', 'electrocuted', 'emergency', 'end', 'engulfed', 'evacuate',
       'evacuation', 'even', 'ever', 'every', 'everyone', 'exploded',
       'explosion', 'eyewitness', 'face', 'failure', 'fall', 'falling',
       'family', 'famine', 'fans', 'far', 'fast', 'fatal', 'fatalities',
       'fatality', 'fear', 'fedex', 'feel', 'few', 'fight', 'film',
       'find', 'fire', 'fires', 'first', 'flames', 'flood', 'flooding',
       'floods', 'food', 'for', 'forest', 'found', 'free', 'from', 'fuck',
       'full', 'game', 'get', 'getting', 'girl', 'give', 'god', 'goes',
       'going', 'good', 'got', 'government', 'great', 'green', 'group',
       'had', 'hail', 'half', 'hand', 'happened', 'hard', 'has', 'hate',
       'have', 'having', 'hazard', 'hazardous', 'head', 'health', 'hear',
       'heard', 'heart', 'heat', 'help', 'her', 'here', 'high', 'hijack',
       'hijacker', 'hijacking', 'him', 'his', 'history', 'hit', 'home',
       'hope', 'horrible', 'hostage', 'hostages', 'hot', 'hour', 'hours',
       'house', 'how', 'huge', 'hurricane', 'image', 'info', 'injured',
       'injuries', 'injury', 'inside', 'instead', 'insurance', 'into',
       'iran', 'isis', 'islam', 'its', 'just', 'keep', 'kids', 'kill',
       'know', 'lab', 'land', 'landslide', 'large', 'last', 'latest',
       'least', 'leave', 'left', 'let', 'life', 'light', 'lightning',
       'like', 'liked', 'line', 'literally', 'little', 'live', 'lol',
       'london', 'longer', 'look', 'looks', 'lost', 'loud', 'love', 'low',
       'mad', 'made', 'make', 'making', 'man', 'many', 'market', 'mass',
       'may', 'media', 'men', 'middle', 'military', 'moment', 'money',
       'more', 'morning', 'most', 'move', 'movie', 'much', 'mudslide',
       'murder', 'murderer', 'must', 'national', 'natural', 'nearby',
       'nearly', 'need', 'never', 'new', 'news', 'next', 'night', 'not',
       'nothing', 'now', 'nuclear', 'obama', 'off', 'official', 'oil',
       'old', 'omg', 'one', 'only', 'order', 'other', 'others', 'our',
       'out', 'outside', 'over', 'own', 'pandemonium', 'park', 'part',
       'past', 'peace', 'people', 'person', 'phone', 'photo', 'photos',
       'pic', 'place', 'plan', 'plans', 'please', 'police', 'poor',
       'possible', 'post', 'power', 'ppl', 'pray', 'prebreak', 'problem',
       'property', 'public', 'put', 'quarantined', 'rain', 'rainstorm',
       'reactor', 'read', 'ready', 'real', 'really', 'reason', 'red',
       'reddit', 'refugees', 'released', 'remember', 'rescue', 'rescued',
       'responders', 'right', 'riot', 'rioting', 'rise', 'river', 'road',
       'rock', 'rubble', 'run', 'running', 'said', 'sandstorm', 'save',
       'saw', 'say', 'says', 'school', 'second', 'security', 'see',
       'seismic', 'send', 'service', 'services', 'set', 'she', 'ship',
       'shit', 'should', 'sign', 'since', 'sinkhole', 'sinking', 'sirens',
       'sky', 'smoke', 'snowstorm', 'some', 'someone', 'something',
       'sound', 'state', 'stay', 'still', 'stop', 'storm', 'story',
       'structural', 'summer', 'sunk', 'support', 'survive', 'survived',
       'survivors', 'take', 'taken', 'taking', 'tell', 'terrorism',
       'terrorist', 'texas', 'than', 'thanks', 'that', 'the', 'their',
       'them', 'then', 'there', 'these', 'they', 'thing', 'things',
       'think', 'this', 'those', 'though', 'thought', 'thousands',
       'through', 'thunder', 'till', 'time', 'times', 'today', 'told',
       'tomorrow', 'tonight', 'too', 'top', 'tornado', 'totally', 'town',
       'traffic', 'tragedy', 'train', 'transport', 'trapped', 'trauma',
       'truck', 'true', 'truth', 'tsunami', 'twitter', 'two', 'under',
       'united', 'until', 'update', 'usa', 'use', 'used', 'very', 'via',
       'video', 'view', 'violent', 'wake', 'want', 'war', 'was', 'watch',
       'watching', 'water', 'wave', 'waves', 'way', 'weapon', 'weapons',
       'weather', 'week', 'well', 'went', 'were', 'what', 'when', 'where',
       'which', 'while', 'whirlwind', 'white', 'who', 'whole', 'why',
       'wild', 'will', 'wind', 'windstorm', 'with', 'without', 'woman',
       'women', 'work', 'world', 'worst', 'would', 'wounded', 'wounds',
       'wow', 'wreck', 'yeah', 'year', 'years', 'yet', 'you', 'your',
       'youtube', 'zone']
def remove_stopwords(tweet):
    cleaned_text = ' '.join([word for word in str(tweet).split() if word not in stop_words])
    return cleaned_text
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    return [lemmatizer.lemmatize(w) for w in text]


# In[40]:


def clean_tweets(tweets):
    expand_colloquialisms(tweets) #expand common slang contractions
    expand_contractions(tweets)
    demojize(tweets)
    emojis(tweets)
    tweets = re.sub(r'#\w+','',tweets)
    tweets = re.sub(r'@\w+|@[^\s]+','',tweets)
    remove_html_text(tweets)
    remove_url_text(tweets)
    tweets = re.sub(r'\d+', '', tweets)
    tweets = tweets.lower()
    tweets = re.sub(r"\s+"," ",tweets).strip()
    teeets = tweets.lstrip(' ')
    tweets = re.sub(r'\b\w{1,2}\b','',tweets)
    tweets = re.sub(r"\s+"," ",tweets).strip()
    remove_stopwords(tweets)
    tweeet_tokenizer = TweetTokenizer()
    tweets = tweeet_tokenizer.tokenize(tweets)
    lemmatizer = WordNetLemmatizer()
    tweets = lemmatize(tweets)
    tfidf_vectorizer = TfidfVectorizer(max_df = 0.25, ngram_range = (1,2))
    tweets = tfidf_vectorizer.fit_transform(tweets)
    return tweets


# In[41]:





# In[ ]:





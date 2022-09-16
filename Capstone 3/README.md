# Natural Language Processing: Binary Classification of Tweets

## Overview
For my capstone project I decided to apply all that've learned about Natural Language Processing. I'm sure I'm not alone in saying that finding a question that is nuanced enough to give me the chance to show off my data science skills whilst being interesting and unique to me was the hardest part of the process. From Towards Data Science blogs to the many books out there, ou'll find give such conflicting informaiton about whether you should pick a dataset from Kaggle or whether you should webscrape or get it from a companys' API directly to show off your skills, or even use data from your Fitbit. So, I figured rather than having a project that does it all, I would rather apply NLP on something that I find interesting and relevant. 

### So, why classification of tweets? 

Currently, my family lives in four countries across three continents. Whevever a notification about disasters or accidents props up, I know I'm not alone in checking for detailed information to see if my loved ones are safe. So, for my capstone project I wanted to create a classification model which could distinguish between tweets about disasters and those that are not. Furthermore, as I am starting out in my data science journey, I wanted to try out my recently acquired skills in natural language processing. Classification models for NLP is becoming more and more important as we've seen many social media sites implementing fact checkers during recent elections around the world and during the COVID-19 pandemic to help the general public distinguish between real and conspirary information. Maybe we could use the model to develop an app or even a chatbox where users can feed in their tweet of choice and the model can help determine if this is about a disaster or not. But, for now, I  am going to limit the scope of this project to developing the classifcation models using machine learning and deep learning. 

### Kaggle Disaster Tweets
For this project, a dataset of 7631 tweets in English from Kaggle was used with the goal of classifying the tweets into two groups: Tweets regarding disasters and those that are not related to disasters. Definetely check out the dataset here: https://www.kaggle.com/competitions/nlp-getting-started/data.

## Data Wrangling and Exploratory Data Analysis 

### The Data 
The dataset included 5 columns: id, keyword, location, tweets, and label (target) of 1 or 0. All 7631 entries had texts and their labels, however 33% of the location entries and 0.80% of the keyword entries were missing. The dataset was noted to be unbalanced with 4342 non-disaster tweets and 3271 disaster tweets. The bar chart on the right highlights the distribution of the data for the two different target classes. Target class 1 refers to real disasters and 0 for non disaster tweets.

 
![Alt text](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/Distibution_of_data.png)

The id numbers were not ordered so I decided to dropped this column. Then, I focused on the keywords and the locations of the tweets to see if these could yield any divisive information on the two classes. As less than 1% of the keywords were missing, I was hoping this could provide some early insight into the two classes. I was also hoping the large chunk of entries missing for location could provide an insight as well.

### Keywords
Of the 7631 entries in the dataset, only 7552 of them had keywords of which only 221 of them were unique. I wanted to explore these 221 keywords, and see if they could help distinguish between the two classes. 

To start with, I had a quick look through the keywords. The first things I noticed was there were plenty of punctuation and accented letters in them, so this gave me an idea of how to start the cleaning process. I initially created a wordcloud and found a few common expressions such as '%20', hashtags, and '%' signs were the ones that popped up quite a bit. I used regular expressions alongside the unidecode library to first wrangle the keywords.

I, then, wanted to look at a countplot of the words. This was a little tricky. My goal was to see if I could group the keywords into the two classes, and then see how often they were used in the two classes. So, first I grouped the keywords by their target classes, and then I used the .transform() method to work out the mean count of the keywords in the respective classes.

![Distribution of Keywords before Stemming](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/keywords_distributions_before_stemming.png)

What I found was a little disappointing but not surpising. Only a couple of words were exclusively used in one class over the other. Such as derailment, debris, and wreckage only seemed to appear in disaster tweets whilst aftershok only appeared in non-disaster tweets. 4 words is not really enough to form a model which is disappointing but not the end of the world. A little further digging did show us some interesting features of our countplot: derailment and derailed for example appears more than once. A couple of words such as 'evacuate' and 'evacuated' and 'suicide bombing' and 'suicide bomber' appeared as independent words and phrases. This suggested I need to consider stemming or lemmatization of the keywords before we any further analysis. 

#### Picking between Stemming and Lemmatization
What is stemming and lemmatization? I have a feeling most people in the 21st centry have seen the benefits of these two types of algorithms even if they aren't directly aware of it: for example, when you search for something on Google, whether you misspell something or when you see lots of different results becuase of the similar names come up. 

1. Stemming is when the last couple of letters, the suffixes, are removed from a word in the hopes of getting to the root word. For example for 'derailed' and 'derailment' can both be reduced to 'derail'. 

2. Lemmatization is when we break down the word into essentially the form we can find in a dictionary. This is more complex than stemming as lemmatization takes into account the contextual use of the words too. 

To understanding the difference between the two processes check our this link: https://www.sinequa.com/guide/natural-language-processing-guide/

Initially I was reluctant to stem the keywords as too much cleaning can sometimes run the risk of losing the sentiments of the texts. However, when I had a look through the keywords (have a look at the chart below) I realised the risks of that was quite low with this dataset. I went with stemming over lammetization because keywords are not really sentences, so the words/phrases didn't really have contextual meaning. So, I used PorterStemmer to further clean up the keywords. 

If we take a look at the chart above with the one below, we can see that words like evacuated has been reduced to their stems 'evacu'.
![Distribution of Keywords After Stemming](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/keywords_distributions.png)

It is worth noting that even after the cleaning and stemming, we aren't close to having the perfectly clean dataset. For exmaple, 'wild fir' and 'wildfir' are both referring to wildfires, and bleed and blood could benefit from combining as well. The chart is also useful in highlighting one of the most common limitations of stemming: we see words that are reduced to 'roots' that aren't actually words nor roots in the English language. 

I then created wordclouds to see the most most common keywords in the two different classes. This was also a helpful visualisation method which helped identify what characters were appearing in the keywords that I wanted to include in my regular expressions during the cleaning process. 

#### Disaster Keywords
![Disaster Keywords](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/wordcloud_disaster_keywords.png)

#### Non Disaster Keywords
![Non Disaster Keywords](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/wordcloud_non_disasterwords.png)

There were some commonalities in the choice of keywords but we also see there are keywords that appear more frequently in one class compared to others: body bags for example appears a lot more frequently as a keyword for non disaster tweets than in disaster tweets. There are a couple of words, debris and wreckage, which only appear as keywords for disaster tweets whilst aftershock, for example, only appears as a keyword in non disaster tweets. As the sample of keywords that cna be used to exclusively distinguish one class from another, we can conclude that there aren't sufficient differences in the keywords to try and develop a classification model. As a result we will drop this for future analysis.

### Location
Initially I had thought of looking at the missing data: maybe one class was missing location data more than the other. However, roughly 33% of the location data was missing from the dataset: 32.9% of tweets in the disaster class are missing and 33.6% of the tweets in the non-disaster classes are missing. This highlights that by looking for what data is missing won’t be very helpful either as the distribution of missing data is equal amongst the two groups. There are 3341 unique location tags in the dataset, which included dates and not always a city or a country. For example 'Est. September 12' and "AFRICA'. Punctuation such a as quotation marks, hashtags frequently appeared when I had a quick look through the data. Furthermore, some entries were more detailed than others. Some included the city and the country whilst some just the city or the country: Vancouver, Canada and Lincoln. This led me to think perhaps the location tags is going to be less useful that I'd like: there are lots of places called Lincoln in many different countries in the world. I also realised that I was going to have to clean up the location tags through a similar method to the keywords. So, again using regular expressions, I removed punctuations, accents, numerical characters, and finally stripped the locations of extra white spaces using the .strip() method. This reduced the unique entries from 3341 to 3106. 

I thought about looking at the similarity of the words: is there a similarity between locations of tweets about disasters and non disasters.I used spaCy's small English library to the identify the locations and worked out the common locations between the two classes. Out of the 3106 unqique locations and 5080 total entries, there were only 82 locations that commonly appeared between the two classes. This is why I tried out a cosine similarity using Scikit-Learn. I got a score of 0.239, highlighting the words in the two classes were not very similar. This is definately worth exploring further but at this point I didn't really know how to further look into this. So, exploring the low similarity score is something I am going to come back to in another project. 

### Tweets

Finally, it was time to look at the tweets! Just like with the keywords, I wanted to be a little weary of cleaning up the tweets. This is why before any cleaning of the the tweets I extracted the number of hashtags, mentions, emojis, and emoticons because we would lose this information when we strip the tweets of punctuation and special characters.

#### Hashtags
 
Majority of the tweets used 1-2 hashtags across the two classes, but disaster tweets have a higher mean with a slightly higher standard deviation. Inspection of the hastags used also highlighted some with greater than 11 hastags: a third of which were related to non disasters and two thirds to disasters. This supports the summary statistics that the disaster tweets to have a slightly higher use of hashtags. 

![Hashtags](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/Num_hashtags.png)

Both the classes show a right skewed distribution. As a result, I concluded removing the hastags wouldn't result in losses of valuable information for our sentimental analysis later on. 

#### Mentions

This showed a very similar right skewed distribution as the hashtag use wiith the modal use for both classes being 0 mentions. So, I decided to remove '@' before further analysing the tweets. 

![Mentions](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/Num_mentions.png)

#### Emojis and Emoticons

I used the emjoi2.0.0 library to 'demojize' the tweets. Honestly, I found this a little challenging and here's a link I found to be pretty helpful: https://medium.com/analytics-vidhya/some-handy-functions-for-text-cleaning-and-manipulation-42bece1f390b

The range of emojis and emoticons use in both the classes were 0-1, with the modal use being 0. Both of them showed the same 
![Emojis](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/num_emojis.png)

Again, the use of emmojis had a right skewed distribution for both the tweets, so I dropped these as well during the cleaning process. 

#### Cleaning tweets

Now comes the most interesting part of the cleaning process. I'd read up on quite a few articles and papers before I dived into cleaning the tweets. Here are a few concerns I had: 

1. I know we want to remove punctuation, but people use punctuation to express feelings. For example !!! vs no exclamation marks at all can convey enthusiasm, fears, and many other sentiments. 


2. Colloquialisms and contractions are are commonly used in Twitter given the 280 characters you're allowed, and they are also a large part of the language used in social media. The character limit and how it affected the use of language is tweets was cited as one of the many reasons for Twitter doubling the character limit from 140 to 240 in 2017. So, how I deal with colloqualisms and contractions was going to be important. Firstly, I realised pretty early there is no 'perfect' way of doing this. A lot of digging through the internet gave me the idea of coming up with an dictionary of common contractions, but this also was going to be time consuming. Then, I hit jackpot! I came across a github repo that already had a pretty extensive list of common contractions- check this out: https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt

I also came across the library called contractions which also was pretty helpful in expanding the other contractions in the tweets further. One word of advice, always expand the contractions before remove anything from the text. This is because if you remove punctuation or lowercase the text first, certain contractions aren't picked up on. So, after all of the words were expanded, the numerical values, html texts, urls, hashtags, mentions, accents were removed and all of the tweets were converted to lowercase. Then, numerical features such as the number of words, number of characters, average length of words were all extracted. 

#### Features from the tweets

The distribution of number of characters and the number of words seems to be bimodal for disaster tweets. This is not surprising as the number of words and the number of characters have a strong correlation between them. The average lenght of the words seem to be shorter for disaster tweets that those not about disasters, and we don't see the same bimodal trend.

For both classes the number of characters in the tweets have a left skewed distribution unlike for the number of words which is somewhat symmetrical. The vertical lines in the graphs below illustrate the mean of the different classes for two classes for the number of characters, number of words, and the mean length of the words. 

![Number of characters](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/Num_characters.png)

![Number of words](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/Num_words.png)  ![Length of words](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/mean_word_length.png)

I concluded that the distributions are so similar for the two classes that these features aren’t very useful for the classification models I want to build. 

#### Stopwords
My next focus was on identifying and removing stopwords. As a dataset of 7000 is not really very large, I decided to not use standard stopwords as this might drastically reduce the size of the corpus. So I ceated my own list of stopwords. 

Stopwords are words that are commonly used in language that are necessary for sentence/phrase construction but not necessarily important for describing the content. So let's say we have an article that describes the types of stars that exist in the universe. Here, the word 'star' is most likely to be frequently used but won't really be useful in differentiatng between the different types of starts discussed. So, here 'star' would be a stop word. For tweets this could include articles and pronouns. 

To come up with my list, I used the CountVectorizer library from Scikit-Learn. This is where the words in the text are vectorised depending on how many times they are used in the text. There were a couple of interesting things that this showed me. 

1. Firstly, I realised how my cleaning process wasn't quite perfect even after I'd spend ages trying to identify patterns. There were plenty of woords like 'aaaaaand' and 'aace'. So I added these to the list of words I'd like to omit. 

2. Lot's of giberish words were still in the corpus such as 'abolxmhvy'. 

For the sake of my curiosity, I also tried out the TF-IDF vectoriser as a couple of the DataCamp courses I'd takes had their instructors mention how this was the vectorizer for their choice. It's a similar method but instead of forming a list based on frequency, TF-IDF stands for term frequency - inverse document frequency. It essentially measures how mnay times a word appears in a document and the inverse document freqeuncy of the word across the corupus. The higher the score the more relevant the word is to the corpus. As expected, I found pretty similar words were listed as those with low scores as the count vectorizer method. However, there were a few differences. So, I decided to create an updated list of stopwords combining 100 words from the CountVectorizer list and the TF-IDF list and removed them from the corpus. 

Once the corpus was cleaned, we looked at the use of nouns and proper nouns for the two different groups. Both classes had a very similar distribution for both proper nouns and nouns in general. 

![Number of proper nouns](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/Num_proper_nouns.png)

![nouns](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/Num_nouns.png)

#### Correlation

Now, the tweets had their numerical features extracted I wanted to see if there were any strong correlations between any of these features. If there were then of course one of them can be dropped before getting this information ready for our machine learning models. Below is a heatmap to visualise the correlation between these features. 

![heatmap](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/heatmap_correlation.png)

Unfortunately, the numerical features showed very little correlation between each other.

## Preprocessing Tweets

### Tokenization and Lemmatization

I find tokenization, or rather the problems and limitations with tokenizations in natural language processing to be fascinating. I did spend quite a bit of time going through DataCamp courses and reading blogs and articles on this before actually tokenizing my corpus. 

#### What is interesting about tokenization?

Well tokenization is when we break down sentences, and essentially the corpus, into smaller chunks, aka tokens. This allows us to change our natural language data set into numerical features needed for machine learning or deep learning models. Essentially, it 'helps' machines to understand words as part of the whole corpus better. Actually, there seems to be even greaer uses for tokenization in NFTs and cybersecurity but that's well outside the scope of this project. 

So what and why are there liminations? Well, us humans can usually look at a word or symbols and figure out context pretty easily. For example, $100 and £100 we cna easily just differentiate as currencies. Machines find even this distinction harder to do. Now, if we remove the currency symbols, understanding the meaning of the 100 becomes even harder. As a concequence there are lots and lots of different ways we can tokenize a corpus. I went with the TweetTokenizer because I was working with tweets. So pretty much all of the hardwork was done for me! Definately feel free to check out the documentation here: https://www.nltk.org/_modules/nltk/tokenize/casual.html

#### Lemmatization vs Stemming
In this case, I went with lemmatization of the tweets rather than stemming as context would be important here. I used the WordNetLemmatizer() from NLTK mainly because of a DataCamp course which raved about this lemmatizer, but there are plenty more out there that could be used here. 

## Model Development and Evaluation

### Machine Learning Models for the Binary Classificaiton of Tweets

Now the fun part: creating the mahcine learning models. First I had to decide which ones I'd like to use, and then optimize each of the parameters of the models so for we can find the 'best' model for this classification problem. Then, of course, I needed to come up with a way of evaluating which model is the 'best'. In my previous project on the course, the classification of breast cancer tumours, it was best to pick a model that had higher flase positive rates than false negative rates. As in healthcare, medical professionals don't just use one test to make a diagnosis. In the case of NLP, the preferred evaluation metric was a little harder to decide on. 

First up was optimising our models. As I'd mentioned earlier we needed to decide on how to break up our tweets into smaller blocks whilst keeping contextual meaning: is it better to look at each separate word, lemmas, or is it better to groupe them into pairs, or threes which is known as the ngrams range? I used a baseline model, the Multinomial Naive-Bayes Classifier, alongside GridSearchCV from Scikitt-Learn to come up with an answer to these questions.

#### Why Multinomial Naive Bayes?
It's a really popular algorithm used in NLP that's used for text data analysis for datasets that have multiple classes. As my goal here was to set up the optimal conditions for my vectorisation, this easy implementation of the multinomial Naive Bayes classifier made it an obvious choice. I've read that a disadvantage to this classifier is that the accuracy of its predictions can be a little lower than other probability algorithms, but as I wanted to use this as a baseline I decided the advantages of it's simplicity outweighed the disadvantage. 

#### Vectorizer
Previously, during the cleaning process I'd explored the TF-IDF Vectorizer and the CountVectorizer. So I went with the TF-IDF Vectorizer here. As tweets are quite short, and my reading, I went with the ngrams range as (1, 1), (1, 2) and (1, 3). The goal is to use GridSearch CV to determine the optimal ngrams range. 





The tweets were split into 80:20 ratio for training and validation. For vectorization, the TF-IDF Vectorizer was used with parameters max_df = 0.25 and ngram_range(1,2) which was determined through GridSearchCV. 

For the baseline model, a Naive Bayes Classifier was used, and then 5 different classification models were selected. For each of these GridSearch with 10 fold cross validation was done to determine the optimal parameters. Then, the best parameters were selected and the performance of the best of each classification model were recorded in the table below. 

The table is arranged in descending orders of accuracy, the area under the receiver operating characteristic curve (AUC_ROC), and the Matthew’s correlation coefficient (MCC) as this is a classification model. As a result, logistic regression was determined as the best machine learning classification model. 

The heatmap highlights the classification of the Logistic Model with the parameters as stated in the table, and the figure on the right is the ROC curve. 

Whilst an accuracy of 79% isn’t necessarily bad, and is definitely an improvement to the logistic model prior to hyperparameter tuning, if another model would be more suitable was explored. So, a Deep Learning model using word embedding was developed. 


### Deep Learning Model with Word Embedding 

Google’s Word2Vec was used to create a deep learning model. This was selected to see if we could keep the semantic closeness of the words in the tweets- think about this as the order at which the words appear in the tweets. The size of the features were trialed with 100, 200, and 300, with 200 giving us the highest accuracy of 61% . 

For the neural networks there were three layers, the first one with 512 neurons, and two subsequent ones with 256 neurons each. A final layer was added, the output layer, with 1 neuron. Whilst, theoretically there are many different activation functions that can be used, for the first three layers Rectified Linear Unit function was used. ReLU function is one of the most popular activation functions and was selected because it is so much more computationally efficient compared to the rest. For the final layer the sigmoid function was used for activation as this is a binary classification and we are looking for outputs of whether a tweet is a disaster or not.

A dropout value of 0.2 was used for each layer- as the accuracy of the models hovered between 57-60% there wasn’t a real concern with overfitting of the data. It’s worth mentioning that the data set is relatively small as a result the effects of the dropout is not as evident compared to larger and longer networks. The Adam optimizer was used, again for it’s faster computation time and to reduce parameter tuning. As this is a binary classification model, the loss function of choice was binary cross entropy. 

### Model Selection

When it comes to selecting between the deep learning and machine learning model, there are a few factors to consider. Machine learning requires far less computing power than deep learning so for the purpose of classification the logistic regression model is great. The higher accuracy score is also an added advantage. However, if we are looking to develop this project further to a larger scale, then the deep learning neural network is signfinicatly superior. For example, if we wanted to continuously build upon this model then the neural network’s self learning abilities is a huge advantage. If we could double or triple the current dataset, then the artificial neural network, even at 60% accuracy would be a better choice. 

## Future Work

It would be interesting to further explore the types of words that people use with hashtags, specifically to conduct a parts-of-speech analysis. For example, is it more common to hashtag verbs in a natural disaster such as ‘burning’ or hashtag a proper noun such as ‘EMS’.  For the locations similarly, it would be interesting to explore the geopolitical entities being tagged. Whilst our dataset only has tweets in English, even this dataset can provide an insight into user behavior. Such as, do users in one location always tag their locations or not. This was a little outside the scope of this project as our aim was to look at the tweets themselves rather than understand user behaviour. Another potential strategy would be to make use of the location and keywords data to help develop the classification models. One possible method of achieving this could be to merge the keywords and location into the tweets and form the corpus including all three fields. This may show improvements for our logistic model as it’s a bag of words model. However, the word embedding model, which relies on the pattern the words appear in, might degrade in accuracy as merging these two fields would influence the sentence structure of the tweets. 

Future works to consider to further the scope and findings of this project would be to optimize the deep neural network. Alternatives such as LSTM should also be tried to see if they are a better fit and can improve the accuracy. Furthermore, the effects of using Leaky ReLU for the activation function would be worth exploring further too. For a neural network of this size, I don’t think we’ll see significant improvements with Leaky ReLU but it is worth checking to be sure. 

Whilst the scope of the project was to develop a model that would classify tweets into disaster and non disaster tweets, it is worth exploring the advantage of implementing a chat box function. This could allow individuals to check when they come across a tweet in question to whether it’s a play on words or actually referring to a natural disaster. During this current Russian-Ukrainian conflict we’ve seen Twitter and Reddit emmerge as leading sources of current news, so a chatbox might actually be welcomed by Twitter users.

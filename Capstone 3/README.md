# Natural Language Processing: Binary Classification of Tweets

## Overview
For my capstone project I decided to apply all that've learned about Natural Language Processing. I'm sure I'm not alone in saying that finding a question that is nuanced enough to give me the chance to show off my data science skills whilst being interesting and unique to me was the hardest part of the process. From Towards Data Science blogs to the many books out there, ou'll find give such conflicting informaiton about whether you should pick a dataset from Kaggle or whether you should webscrape or get it from a companys' API directly to show off your skills, or even use data from your Fitbit. So, I figured rather than having a project that does it all, I would rather apply NLP on something that I find interesting and relevant. 

### So, why classification of tweets? 

Currently, my family lives in four countries across three continents. Whevever a notification about disasters or accidents props up, I know I'm not alone in checking for detailed information to see if my loved ones are safe. So, for my capstone project I wanted to create a classification model which could distinguish between tweets about natural disasters and those that are not. Furthermore, as I am starting out in my data science career, I wanted to try out my recently acquired skills in natural language processing. Classification models for NLP is becoming more and more important as we've seen in social media sites implementing fact checkers during recent elections around the world and the COVID-19 pandemic to help the general public distinguish between real and conspirary information.Maybe we could use the model to develop an app or even a chatbox where users can feed in their tweet of choice and the model can help determine if this is about a disaster or not. But, for now, we are going to limit the scope of this project to developing the classifcation models using machine learning and deep learning. 

### Kaggle Disaster Tweets
For this project, a dataset of 7631 tweets in English from Kaggle was used with the goal of classifying the tweets into two groups: Tweets regarding disasters and those that are not related to disasters. Definetely check out the dataset here: https://www.kaggle.com/competitions/nlp-getting-started/data.

## Data Wrangling and Exploratory Data Analysis 

### The Data 
The dataset included 5 columns: id, keyword, location, tweets, and label (target) of 1 or 0. All 7631 entries had texts and their labels, however 33% of the location entries and 0.80% of the keyword entries were missing. The dataset was noted to be unbalanced with 4342 non-disaster tweets and 3271 disaster tweets. The bar chart on the right highlights the distribution of the data for the two different target classes. Target class 1 refers to real disasters and 0 for non disaster tweets.

 
![Alt text](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/Distibution_of_data.png)

The id numbers were not ordered so I decided to dropped this column. Then, I focused on the keywords and the locations of the tweets to see if these could yield any divisive information on the two classes. As less than 1% of the keywords were missing, I was hoping this could provide some early insight into the two classes. I was also hoping the large chunk of entries missing for location could provide an insight as well: perhaps one class uses location tags more frequently than others? Or perhaps those with missing keywords had locations present?

### Keywords
Of the 7631 entries in the dataset, only 7552 of them had keywords of which only 221 of them were unique. I wanted to explore these 221 keywords, and see if they could help distinguish between the two classes. 

To start with, I had a quick look through the keywords. The first things I noticed was there were plenty of punctuation and accented letters in them, so this gave me an idea of how to start the cleaning process. I initially created a wordcloud and found a few common expressions such as '%20', hashtags, and '%' signs were the ones that popped up quite a bit. I used regular expressions alongside the unidecode library to first wrangle the keywords.

I, then, wanted to look at a countplot of the words. This was a little tricky. My goal was to see if I could group the keywords into the two classes, and then see how often they were used in the two classes. So, first I grouped the keywords by their target classes, and then I used the .transform() method to work out the mean count of the keywords in the respective classes.

![Distribution of Keywords before Stemming](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/keywords_distributions_before_stemming.png)

What I found was a little disappointing but not surpising. Only a couple of words were exclusively used in one class over the other. Such as derailment, debris, and wreckage only seemed to appear in disaster tweets whilst aftershok only appeared in non-disaster tweets. 4 words is not really enough to form a model which is disappointing but not the end of the world. A little further digging did show us some interesting features of our countplot: derailment and derailed for example appears more than once. A couple of words such as 'evacuate' and 'evacuated' and 'suicide bombing' and 'suicide bomber' appeared as independent words and phrases. This suggested I need to consider stemming or lemmatization of the keywords before we any further analysis. 

#### Picking between Stemming and Lemmatization
> Stemming is when the last couple of letters are removed from a word in the hopes of getting to the root word. For example for 'derailed' and 'derailment' can both be reduced to 'derail'.
> Lemmatization is when we break down the word into essentially the form we can find in a dictionary. This is more complex than stemming as lemmatization takes into account the contextual use of the words too. 

To understanding the difference between the two processes check our this like: https://www.sinequa.com/guide/natural-language-processing-guide/

Initially I was reluctant to stem the keywords as too much cleaning can sometimes run the risk of losing the sentiments of the texts. However, when I had a look through the keywords (have a look at the chart below) I realised the risks of that was quite low with this dataset. So, I used PorterStemmer to further clean up the keywords.



If we take a look at the chart above with the one below, we can see that words like evacuated has been reduced to their stems 'evacu'.
![Distribution of Keywords After Stemming](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/keywords_distributions.png)

It is worth noting that even after the cleaning and stemming, we aren't close to having the perfectly clean dataset. For exmaple, wild fir and wildfir are both referring to wildfires, and bleed and blood could benefit from combining as well. 

The two word clouds illustrate the most common keywords in the two different classes. 

![Disaster Keywords](https://github.com/iban121/Springboard/blob/main/Capstone%203/notebooks/figures/wordcloud_disaster_keywords.png)

There were some commonalities in the choice of keywords but we also see there are keywords that appear more frequently in one class compared to others: body bags for example appears a lot more frequently as a keyword for non disaster tweets than in disaster tweets. There are a couple of words, debris and wreckage, which only appear as keywords for disaster tweets whilst aftershock, for example, only appears as a keyword in non disaster tweets. As the sample of keywords that cna be used to exclusively distinguish one class from another, we can conclude that there aren't sufficient differences in the keywords to try and develop a classification model. As a result we will drop this for future analysis.

### Location
Initially I thought of looking at the missing data: maybe one class was missing location data more than the other. However, roughly 33% of the location data was missing from the dataset: 32.9% of tweets in the disaster class are missing and 33.6% of the tweets in the non-disaster classes are missing. This highlights that by looking for what data is missing won’t be very helpful as the distribution of missing data is equal amongst the two groups. So, I thought about looking at the similarity of the words: is there a similarity between locations of tweets about disasters and non disasters. This is why I tried out a cosine similarity using Scikit-learn. I got a score of 0.239, highlighting the words in the two classes were not very similar. This is definately worth exploring further but at this point I didn't really know how to further look into this. So, exploring the low similarity score is something I am going to come back to in another project. 

### Tweets

Finally, it was time to look at the tweets!. Just like with the keywords, I wanted to be a little weary of cleaning up with tweets. This is why before any  cleaning of the the tweets I extracted the number of hashtags, mentions, emojis, and emoticons because we would lose this information when we strip the tweets of punctuation and special characters.

For the number of hashtags, mentions, emojis and emoticons, all of the data are left skewed and the majority of the tweets only use one. The distributions were essentially identical for both classes so these features were considered to be not crucial in the classification of the two classes. 

As the language used in tweets can vary greatly in formality, in order to clean the text first contractions had to be expanded. This was done in two ways: first contractions commonly using in colloquial English was expanded upon, and then the library contractions were used for further expansions. Then, the emoji library was used to remove emoticons and emojis from the texts. Finally, numerical values, html texts, urls, hashtags, mentions, accents were removed and all of the tweets were converted to lowercase. Then numerical features such as the number of words, number of characters, average length of words were all extracted. The two figures below show the distribution of the number of words and average length of the words between the two classes. The vertical lines illustrate the mean scores, and so we can conclude that their distributions are so similar they aren’t very useful for the classification. 

Once we had explored the features, we then focused on identifying and removing stopwords. As a dataset of 7000 is not really very large it was decided to not use standard stopwords as this might drastically reduce the size of our corpus. As a result the words that scored the lowest when the Count Vectorizer and TfidfVectorizer were applied were all oimitted: a total of 629 words. 

Once the corpus was cleaned, we looked at the use of nouns and proper nouns for the two different groups. Both classes had a very similar distribution for both proper nouns and nouns (illustrated in the figure to the right) in general. 

The numerical features showed very little correlation and as a result these were decided to be omitted. Instead, the tweets were then tokenized using TweetTokenizer  and then the WordNetLemmatizer was used to lemmatize the tweets so they could be processed further to develop Machine Learning and Deep Learning models. 


## Model Development and Evaluation

### Machine Learning Models for the Binary Classificaiton of Tweets

The tweets were split into 80:20 ratio for training and validation. For vectorization, the TF-IDF Vectorizer was used with parameters max_df = 0.25 and ngram_range(1,2) which was determined through GridSearchCV. 

For the baseline model, a Naive Bayes Classifier was used, and then 5 different classification models were selected. For each of these GridSearch with 10 fold cross validation was done to determine the optimal parameters. Then, the best parameters were selected and the performance of the best of each classification model were recorded in the table below. 

The table is arranged in descending orders of Accuracy, the area under the receiver operating characteristic curve (AUC_ROC), and the Matthew’s correlation coefficient (MCC) as this is a classification model. As a result, logistic regression was determined as the best machine learning classification model. 

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


# References 
Data [https://www.kaggle.com/competitions/nlp-getting-started/data] 

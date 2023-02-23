# Applying the Data Science Method: Developing a Predictive Model for Big Mountain Resort's Pricing Strategy


## Step 1: Problem Identification 
The aim of this step is to synthesize the context and the problems outlined by the client, in this case Big Mountain Resort, into a specific, measureable, attainable, relevant, and time-bound problem statement we will address. 

To synthesize the question I addressed a few criteria: 
1: Context: Its important to first identify the problem we want to solve, but to do so we need to know how the business operates which has resulted in the problem to begin. In this case Big Mountain Resort in Montana has recently had an increase in operation cost of 1.54 Million USD in the last reason as a result of a new chair lift they have installed. While Big Mountain in currently charging a premium rate compared to other ski resorts in the space. Big Mountain Resort does have premium facilities, so they are questioning whether their ticket prices reflect the premium services they offer. 

2. Criteria of Success: This identifies the scope of the problem we are trying to solve. 
In this case Big Mountain recently had an increased operational cost of 1.54 million dollars, so the data driven strategy this project proposes must ensure a profit of at least 1.54 million dollars to address the increased costs. Furthermore, the motivations for the chair lift installation was to attract more visitors, so our proposed solution must help increase the average number of visitors from the current 350,000. 

3. Scope of Solution Space: this helps us narrow down the task even further so we have a more spcific problem to solve. It would be lovely to say we can 'make the resort more profitable' but in what time-frame? What does 'more profitable' actually mean when it doesn't outline by how much we want to increase profits. For this project we want to see the increase in profits of 1.54 million and the increase in the number of visitors by the end of the year. 

4. Constraints within the Solution Space: This is an opportunity to proactively identifying possible problems before they actually occur so we can avoid major issues. For this project, the dataset didn't provide information about the different types of customers they have, nor any idea on what facilities are popular with which customers. Furthermore, snowfall is dependent on weather conditions. Lack of snowfall results in increased costs as the resort might have to provide fake snow in popular routes. Without detailed information about these conditions we will have limitations to what our solution can acheive. Whilst, we can still provide a meaningful insight into a data driven strategy, it's important to note the limitations as well. 

5. Stakeholders: This tells us who can provide insights and also who we will be liasing with. Stakeholders can provide domain knowledge pretaining to the specific needs of the project. Furthermore, knowing who the stakeholders are also allows us to understand how to present out findings. If the stakeholders are the technical team of a company, we need to ensure we can communicate the techincal know-hows effectively to justify our findings. For this project we were given the names and positions of the stakeholders. 

6. Key Data Sources Required? This gives us an idea of what data we have access to, whether they can add more value to the data provided or if they can provide other types of domain knowledge for the project. 

**Synthesized Problem Statement: Implement a data driven business strategy which highlights the current facilities at Big Mountain Resort to increase ticket prices resulting in an increase of profits by at least 1.54 Million USD per season in the coming 12 months.**

## Step 2: Data Wrangling
Initially, I thought this is the stage where you clean up the data, but as I learnt more and more about the various application of data science I realised there is such a thing as cleaning the data too much during data wrangling. It's about finding the sweet spot of where you have cleaned the data sufficiently that you can extract patterns and features from the data given, but you've not cleaned so much that we can't draw conclusions from what is missing and spurious. Sometimes omissions can help us identify patterns and trends too! My biggest takeaway from this project was learning how the data science method can be cyclical at times: it's possible to go around a loop a couple of times between data wrangling and exploratory data analysis to optimize our learning from the dataset. 

For the Big Mountain Resort project, there were three main steps in the data wrangling stage. 

1. Getting the Data: Getting a given dataset from the client isn't enough, but rather getting the right data is important. This is about making sure when necessary we motify the dataframe by sourcing relevant information or sometimes, having to discard some of the data. 
2. Shape and Type of Data: It's important to identify what data is missing and how many of them are missing compared to the other features, and then to decide how to handle the missing information. Sometimes it's perfectly acceptable to drop a feature if there's insufficient datapoints, whilst at other times we might choose to replace the missing values with the median, mode, or the mean values. How we choose to handle missing data therefore is very dependent on the type of data we are dealing with: is the feature categorical? Is it structured or unstructured data? 
For this project, there was only one feature that was dropped, but the data types had to be cleaned up a little. Furthermore, to actually gain a deeper understanding of Big Mountain's performance in the space, state-wise summary tables were created to so we can understand how similar or different Big Mountain is compared to their competitors in Montana. 
3. Quality of the Data: This is where we judge the quality of the data provided. This can be as simple as the validity of the data: are the numbers actually correct? In this case, when looking at spurious entries, this can be any data points that are very weirdly large or small, we realised there was an obvious mistake during data entry. Whilst in this case the problem was easy to fix as we could just google the data, there are plenty of cases we may not be able to do this. In those cases we really need to consider whether the poor quality in the data is actually going to allow us to draw meaningful conclusions or not. In some cases, this stage might be the end of the DSM if additional data cannot be collected. For this project as the goal of this project is to determine a pricing strategy, any entries where prices were missing were dropped. Another aspect to consider is whether the dataset can actually answer the question we set out to investigate at the inception of the project. We want to know whether Big Mountain is pricing their facilities, and as a result their ticket prices, appropriately. As a consequencey, we needed to collect more data on the different states the different ski resorts are located. We choose to supplement the dataset provided with information regarding the population and reas of the states the resorts were located in the United States so we can consider the demand and supply of ski facilities in the region. 
## Step 3: Exploratory Data Analysis

## Step 4: Preprocessing and Modelling

## Step 5: Modelling

## Step 6: Presenting Findings 

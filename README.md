# sentiment-analysis
Just keeping my files here for clout

# general
The model is simple but works for basic sentiment analysis, categories are positive, neutral, and negative. It uses two hidden layers with an input vector of 48554 elements using a BoW approach. The ReLU activation function is used, except for the output layer which uses the sigmoid function. The dataset used was found on Kaggle and consists of about 25000 tweets to the @Dell acocunt, which are labelled by machine. The first 20000 elements were used for training. The total accuracy along the testing set (rest of dataset) is about 0.7, with per category accyracies of:
    positive: 0.6
    neutral: 0.86
    negative: 0.75
where is should be noted that the dataset contains slightly more negative samples than other categories(about 40% of total samples)

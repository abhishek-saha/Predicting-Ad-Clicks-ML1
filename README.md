# Project for Machine Learning 1 course
## Predicting Ad Clicks
----------------------------
## Introduction:
#### Online advertising is a form of marketing which delivers promotional messages to consumers via the Internet. Due to the volume of advertisements being published & huge cost associated with marketing campaigns, serving the right advertisement to the right consumer is extremely important. Hence, the ability to predict whether a consumer will click an advertisement on a search engine such as Google will help the firm determine the right advertisement to serve the right consumer. Our project intends to unravel this prediction by creating a classification model, utilizing data of 9 days (Oct 21st - 29th 2014) during which more than 30 million instances of advertisements were served to various consumers.
-----------------------------
## Models implemented:
We built and implemented models using the following algorithms:
- Lasso logistic regression
- Ridge logistic regression
- Trees
- Bagging
- Random Forest

The parameters were tuned using different parameters to come up with the optimum values. The performance on validation dataset was as follows:
- Lasso regression - Log loss - 0.4065
- Ridge regression - Log loss - 0.4068
- Trees - Log loss - 0.4229
- Bagging - Log loss - 0.8478
- Random forest - Log loss - 1.14

Hence the best algorithm was lasso regression which was used to predict the clicks on the test set.

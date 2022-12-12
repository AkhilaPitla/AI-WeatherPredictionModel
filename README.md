# AI-WeatherPredictionModel\
INTRODUCTION:

The science of forecasting the weather is crucial. Accurate forecasting can lessen property damage and save lives. The ability to track the ideal time to plant or assist farmers in protecting their crops makes it essential for agriculture as well.

And in the upcoming years, its importance will only increase. Because of climate change and variability, severe weather events are occurring more frequently and with greater intensity.

REQUIREMENTS:

•	Codes are run on Windows. It is based on Python 3.6. Required packages are stated below.

•	numpy
•	pandas
•	sklearn
•	Install the packages by below way

•	pip install - <” include the packages that are not available in your system”>


PIPELINE FOR QUICK START:

•	All raw data are in weather.csv, where it is used to split data into Train and Test data.

•	Or you want to download the dataset by yourself, then you can go to https://www.kaggle.com/datasets/zaraavagyan/weathercsv?resource=download

•	Train Models We used Decision Tree Classifier, Logistic Regression and Support Vector Classifier machine learning algorithms to train the data set.


PROJECT OBJECTIVES:

•	In this project, we are predicting the weather by training a dataset with several machine learning algorithms and evaluating the performance of each model.
•	The model that we will create will be based on historical usage rates. 
•	Machine learning models like Decision Tree Classifier, Logistic Regression and Support Vector Classifier.

APPROACH:

1.Retrive the dataset from the API.
2Exploratory Data Analysis.
3.Train the dataset with ML model.
4.Predict the model with test data.
5.Performance evaluation.
DELIVERABLES:

1.	Create the train dataset.
2.	From SkLearn import the ML model.
3.	Perform train dataset on these models.
4.	Predict the model using test dataset.
5.	Evaluate the model using R2score, Root Mean Square Error, etc.,

Machine Learning Models:

A machine learning model is a representation of an algorithm that searches through vast amounts of data to look for patterns or forecast the future. The mathematical powerhouses of artificial intelligence are machine learning (ML) models, which are fed by data.

For instance, a computer vision ML model may be able to recognize autos and pedestrians in a real-time video. To translate words and sentences, one may use natural language processing.

A.	Logistic regression:
Based on some dependent variables, the machine learning classification process known as logistic regression is used to forecast the likelihood of a given class. In essence, the logistic regression model generates the logistic of the outcome after computing the sum of the input features (in most cases, there is a bias term).

For a binary classification problem, logistic regression's result is always between 0 and 1, which is appropriate. The likelihood that the current sample will be assigned to class=1

How it works:

A statistical method known as logistic regression is used to forecast the likelihood of a binary response based on one or more independent factors. This indicates that logistic regression is used to forecast outcomes with two possible values, such as 0 or 1, pass or fail, yes or no, etc., given a set of parameters.

In the image below, you can find the accuracy and evaluation of the logistic regression model with our dataset.

 

B.	Decision Tree:
One of the well-known and effective supervised machine learning methods that may be applied to classification and regression issues is the decision tree. Decision rules that were taken from the training data formed the basis of the model. When solving a regression problem, the model substitutes the value for the class and utilizes the mean squared error as a measure of judgment accuracy.



How it works:
There are three different sorts of nodes in this tree-structured classifier. The root node, which can be further broken into nodes, is the first node that represents the complete sample. Internal nodes stand in for the dataset's attributes and branching for the decision-making processes. The outcome is represented by the leaf node at the end. When trying to solve problems involving decisions, this algorithm is highly helpful.

In the image below, you can find the accuracy and evaluation of the logistic regression model with our dataset. We can observe that this model gives us 100% accuracy.

 

C.	Support Vector Classifier:
A supervised machine learning model called a support vector machine (SVM) employs classification techniques to solve two-group classification problems. An SVM model can classify new text after being given sets of labeled training data for each category.

They offer two key advantages over more recent algorithms like neural networks: greater speed and improved performance with fewer samples (in the thousands). As a result, the approach is excellent for text classification issues, where it's typical to only have access to a dataset with a few thousand tags on each sample.

How Does it Work:
SVM categorizes data points even when they are not otherwise linearly separable by mapping the data to a high-dimensional feature space. Once a separator between the categories is identified, the data are converted to enable the hyperplane representation of the separator. The group to which a new record should belong can therefore be predicted using the features of new data.

In the image below, you can find the accuracy and evaluation of the logistic regression model with our dataset.

 











Conclusion:

In an ideal world, AI and machine learning will enable human forecasters to work more productively, devoting more of their time to informing the general public—or, in the case of private forecasters, their clients—about the implications and repercussions of forecasts. We think the best way to achieve these objectives and increase confidence in computer-generated weather forecasts is through rigorous collaboration between scientists, forecasters, and forecast users.
In this research, we tried demonstrating how machine learning may enhance weather forecasting accuracy by incorporating some algorithms and historical data, allowing us to prevent natural disasters like hurricanes, cyclones, and floods in the future.

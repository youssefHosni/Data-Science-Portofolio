<h1>Power Consumption Forecasting </h1>

<h2>overview</h2>	 

Power consumption forecasting is s a time series data collected periodically, over time. Time series forecasting is the task of predicting future data points, given some historical data. It is commonly used in a variety of tasks from weather forecasting, retail and sales forecasting, stock market prediction, and in behavior prediction (such as predicting the flow of car traffic over a day). There is a lot of time series data out there, and recognizing patterns in that data is an active area of machine learning research!
<h2>Motivation and the problem</h2>	 

Taking the data of the power consumption from 2007-2009, and then use it to accurately predict the average Global active power usage for the next several months in 2010!

<h2>Data</h2>	 
Energy Consumption Data
The data we'll be working with in this notebook is data about household electric power consumption, over the globe. The dataset is originally taken from [Kaggle](https://www.kaggle.com/uciml/electric-power-consumption-data-set) and represents power consumption collected over several years from 2006 to 2010. With such a large dataset, we can aim to predict over long periods of time, over days, weeks or months of time. Predicting energy consumption can be a useful task for a variety of reasons including determining seasonal prices for power consumption and efficiently delivering power to people, according to their predicted usage.
Interesting read: An inversely-related project, recently done by Google and DeepMind, uses machine learning to predict the generation of power by wind turbines and efficiently deliver power to the grid. You can read about that research, in this post.

<h2> DeepAR model</h2>

DeepAR utilizes a recurrent neural network (RNN), which is designed to accept some sequence of data points as historical input and produce a predicted sequence of points. So, how does this model learn?
During training, you'll provide a training dataset (made of several time series) to a DeepAR estimator. The estimator looks at all the training time series and tries to identify similarities across them. It trains by randomly sampling training examples from the training time series.
Each training example consists of a pair of adjacent context and prediction windows of fixed, predefined lengths.
The context_length parameter controls how far in the past the model can see.
The prediction_length parameter controls how far in the future predictions can be made.
You can find more details, in this **[documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_how-it-works.html)**.

<h2>Notebook outline</h2>	 

* Loading and exploring the data
* Creating training and test sets of time series
* Formatting data as JSON files and uploading to S3
* Instantiating and training a DeepAR estimator
* Deploying a model and creating a predictor
* Evaluating the predictor

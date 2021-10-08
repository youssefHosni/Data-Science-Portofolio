# project summary
In this project,  real-life data from Bertelsmann partners AZ Direct and Arvato Finance Solution were used. The data here concerns a company that performs mail-order sales in Germany. Their main question of interest is to identify facets of the population that are most likely to be purchasers of their products for a mailout campaign. As a data scientist i will use unsupervised learning techniques to organize the general population into clusters, then use those clusters to see which of them comprise the main user base for the company. Prior to applying the machine learning methods, i cleaned the data in order to convert the data into a usable form.

## steps
### Step 1: Preprocessing
When you start an analysis, you must first explore and understand the data that you are working with. In this (and the next) step of the project, you’ll be working with the general demographics data. As part of your investigation of dataset properties, you must attend to a few key points:

How are missing or unknown values encoded in the data? Are there certain features (columns) that should be removed from the analysis because of missing data? Are there certain data points (rows) that should be treated separately from the rest?
Consider the level of measurement for each feature in the dataset (e.g. categorical, ordinal, numeric). What assumptions must be made in order to use each feature in the final analysis? Are there features that need to be re-encoded before they can be used? Are there additional features that can be dropped at this stage?
You will create a cleaning procedure that you will apply first to the general demographic data, then later to the customers data.

### Step 2: Feature Transformation
Now that your data is clean, you will use dimensionality reduction techniques to identify relationships between variables in the dataset, resulting in the creation of a new set of variables that account for those correlations. In this stage of the project, you will attend to the following points:

The first technique that you should perform on your data is feature scaling. What might happen if we don’t perform feature scaling before applying later techniques you’ll be using?
Once you’ve scaled your features, you can then apply principal component analysis (PCA) to find the vectors of maximal variability. How much variability in the data does each principal component capture? Can you interpret associations between original features in your dataset based on the weights given on the strongest components? How many components will you keep as part of the dimensionality reduction process?
You will use the sklearn library to create objects that implement your feature scaling and PCA dimensionality reduction decisions.

### Step 3: Clustering
Finally, on your transformed data, you will apply clustering techniques to identify groups in the general demographic data. You will then apply the same clustering model to the customers dataset to see how market segments differ between the general population and the mail-order sales company. You will tackle the following points in this stage:

Use the k-means method to cluster the demographic data into groups. How should you make a decision on how many clusters to use?
Apply the techniques and models that you fit on the demographic data to the customers data: data cleaning, feature scaling, PCA, and k-means clustering. Compare the distribution of people by cluster for the customer data to that of the general population. Can you say anything about which types of people are likely consumers for the mail-order sales company?

## Requirments
* NumPy
* pandas
* Sklearn / scikit-learn
* Matplotlib (for data visualization)
* Seaborn (for data visualization)

## Data used 

Demographic data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
Demographic data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
The data is not provided as it is private data.


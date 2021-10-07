# Finding the best neighborhood to open a new gym

## Introduction
Finding the best neighborhood in Tornoto city to open a new gym, given the demographic and geogrpahic and venues information data. More inforamtion can be found [here](https://github.com/youssefHosni/Finding-the-best-Tornoto-neighborhood-to-open-a-new-gym/blob/main/Project%20report.pdf)

## Data

The dataset used to solve this problem have the following informtion:
* The demogrpahics information:  They are the **total population**, the **15-45 poulation**, the **number of educated people** and the **number of employers** in each neighborhood. 
* The graphical data (lat,long) for each neighborhood is used to get the venues information for each neighborhood from Foursquare API.

The neighbourhoods on the map are shown in the figure below
![neighborhood_map](https://user-images.githubusercontent.com/72076328/109424179-4bbec100-79eb-11eb-9a71-6557010e2ee3.PNG)
From the Foursquare API the number of venues per each neighbourhood and the number of Gym/Fitness centers per each neighbourhood were calculated and then merged with the demographics data and the final data used is as the shown in the figure below.
![total_data](https://user-images.githubusercontent.com/72076328/109424261-a9530d80-79eb-11eb-807c-49864647abc6.PNG)
The final dataset can be found [here](https://www.kaggle.com/youssef19/toronto-neighborhoods-inforamtion)

## Methodology 
### Data preprocessing
The data were normalized using the min-max normalization. This is an important step because the k-means algorithms depend on distance measurement, so it is important that the data used be in a similar scale. The formula of the min-max scaler is as the following: 
`𝑓𝑒𝑎𝑡𝑢𝑟𝑒−min⁡(𝑓𝑒𝑎𝑡𝑢𝑟𝑒)max⁡(𝑓𝑒𝑎𝑡𝑢𝑟𝑒)−min⁡(𝑓𝑒𝑎𝑡𝑢𝑟𝑒)`
The neighborhood and the geographical data were dropped from the data as they will be used by the clustering algorithm. 
### K-means clustering 
The best k was found using the elbow method, in which the average distance from the clusters is calculated for different values of k and the best k is the k at the elbow. The best k was found to be 3.

## Results 
The neighborhoods are clustered into three clusters as shown in the figure below. The red color is the first cluster, the violet is the second cluster, green is the third cluster.
![clsuters on the map](https://user-images.githubusercontent.com/72076328/113056147-18bb4900-91b4-11eb-9e33-8ccf83fa5fca.PNG)

## Conclusion 
Using the demographics data and the venue information for each neighborhood obtained from Foursquare API, I was able to cluster the neighborhoods into three clusters using the K-means clustering algorithm. The number of gyms was found to be correlated to the number of venues. The neighborhoods with a large number of venues and gyms are clustered into the third cluster, so the most suitable neighborhood out of this cluster is the **Trinity-Bellwoods neighborhood**. The first cluster contains neighborhoods with  large population and  small number of gyms and  moderate number of venues. The **Church-Yonge Corridor neighborhood** is the best choice out of this cluster as it contains 98 venues and large population. The number of venues is almost similar to that of the “Trinity-Bellwoods” and the population is double of it, making it the best neighborhood to open a new gym in Toronto city.

## license & Copyright

© Youssef Hosni

Lincesed under [MIT Linces](https://github.com/youssefHosni/Finding-the-best-Tornoto-neighborhood-to-open-a-new-gym/blob/main/LICENSE).

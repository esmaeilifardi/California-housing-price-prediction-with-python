# California-housing-price-prediction-with-python
Welcome to the California Housing Prices Analysis! In this project, we are going to use the 1990 California Census dataset to study and try to understand how the different attributes can make the house prices get higher or lower. How does the location impact? How about the size of the house? The age?

This dataset has a lot of information that can help us. The main goal is to build a Machine Learning Model in python that can learn from this data and make predictions of the price of a house in any district, given all the other metrics provided in the dataset.

The project will be divided in 2 main parts: First, we’ll take a deep dive in the data, clean it up, and make a big EDA to gather insights and create hypotheses that might be helpful to the model.

After that, we’ll jump through hoops to create a ML model capable of making the best possible prediction on the house prices: we’ll test different models, check the hypothesis elaborated, try to create new features, optimize the hyperparameters and so on.
First Look at the Data
Let’s take a look at the raw dataset, collected from the 1990 Carlifornia Census. It contains 20640 rows and each one of them stores information about a specific block.
We have the following variables, with the descriptions below collected in the dataset page in kaggle.

longitude: A measure of how far west a house is; a higher value is farther west
• latitude: A measure of how far north a house is; a higher value is farther north
• housing_median_age: Median age of a house within a block; a lower number is a newer building
• total_rooms: Total number of rooms within a block
• total_bedrooms: Total number of bedrooms within a block
• population: Total number of people residing within a block
• households: Total number of households, a group of people residing within a home unit, for a block
• median_income: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
• ocean_proximity: Location of the house w.r.t ocean/sea
• median_house_value: Median house value for households within a block (measured in US Dollars)
Okay, right away we have a little obstacle here. Our target variable is the “median_house_value”, that is the median value of one house in a given block. However, some of the metrics are related to the whole block (total_rooms, total_bedrooms and population). We might need to create some new features to guarantee that they are in the same “unit”.

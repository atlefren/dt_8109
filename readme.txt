Code for DT8109 Business Systems Case Study 2017
================================================

Code used for the case study in DT8109 Business Systems at NTNU.

Basically does four things:

1. Gathers a lot of data about the dataset (events_stats.py)
2. Performs a hierarcical clustering (cluster_events.py)
3. Performs a classification using naive bayes (segment.py)
4. Performs an analysis of when an event breaks even using tree induction (predict_sold_out.py)


dist.py contains helper funtions for calculating distances, and get_events.py contains utilities for reading data.
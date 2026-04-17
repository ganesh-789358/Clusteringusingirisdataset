This project demonstrates clustering on the Iris dataset using the K-Means algorithm. It analyzes features like sepal length, sepal width, petal length, and petal width to group similar flowers. The dataset preview shows sample values used for training the model and identifying natural clusters in the data.

![](https://github.com/ganesh-789358/Clusteringusingirisdataset/blob/main/Screenshot%202026-04-17%20175119.png)

This visualization shows the Elbow Method used to determine the optimal number of clusters (k) for K-Means. The graph plots WCSS (Within-Cluster Sum of Squares) against different values of k. As k increases, WCSS decreases, but the rate of decrease slows. The “elbow point” (around k=3) indicates the ideal number of clusters for the Iris dataset.

![](https://github.com/ganesh-789358/Clusteringusingirisdataset/blob/main/Screenshot%202026-04-17%20175134.png)

This scatter plot visualizes the K-Means clustering results using sepal length and sepal width. Each point represents a flower, and colors indicate different clusters. The plot shows how the algorithm groups similar data points together, revealing clear separation between clusters based on feature patterns.

![](https://github.com/ganesh-789358/Clusteringusingirisdataset/blob/main/Screenshot%202026-04-17%20175226.png)

This plot shows K-Means clustering with a higher number of clusters (k=9). Each color represents a different cluster, resulting in more fragmented groupings. Compared to smaller k values, the data is over-segmented, making clusters less meaningful. This highlights why choosing an optimal k (like 3) is important for better pattern recognition.
![](https://github.com/ganesh-789358/Clusteringusingirisdataset/blob/main/Screenshot%202026-04-17%20175257.png)

The Iris clustering project shows that K-Means effectively groups similar data points based on flower features. Using the Elbow Method, the optimal number of clusters is around 3, which aligns well with the actual species. Lower or higher k values lead to underfitting or over-segmentation. Overall, the model successfully uncovers natural patterns in the dataset.

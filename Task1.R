# Load required libraries
library(tidyverse)
library(NbClust)
library(cluster)
library(factoextra)

# Load the dataset
wine_data <- read.csv("C:/Users/Asus/Desktop/ML CW/Whitewine.csv")

# Part a: Pre-processing tasks
# Before Removing Outliers
summary(wine_data)

# Scaling the data
scaled_data <- scale(wine_data[, 1:11])

# Detecting outliers
outliers <- boxplot(scaled_data, plot = FALSE)$out

# Removing outliers
wine_data_no_outliers <- wine_data[!apply(scaled_data, 1, function(x) any(x %in% outliers)), ]

# After Removing Outliers
summary(wine_data_no_outliers)

# Part b: Determining the number of cluster centres via automated tools
# NBclust
nb_results <- NbClust(wine_data_no_outliers[,1:11], distance = "euclidean", min.nc = 2, max.nc = 10, method = "complete", index = "all")
nb_results$Best.nc[1]  # Best number of clusters suggested by NBclust

# Elbow method
set.seed(123)
wss <- numeric(10)
for (i in 1:10) {
  wss[i] <- sum(kmeans(wine_data_no_outliers[,1:11], centers = i)$withinss)
}
plot(1:10, wss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")

# Gap statistics
set.seed(123)
gap_stat <- clusGap(wine_data_no_outliers[,1:11], FUN = kmeans, nstart = 25, K.max = 10, B = 50)
plot(gap_stat, main = "Gap Statistic Plot")

# Silhouette method
sil_width <- c(NA)
for (i in 2:10) {
  km <- kmeans(wine_data_no_outliers[,1:11], centers = i)
  sil_width[i] <- mean(silhouette(km$cluster, dist(wine_data_no_outliers[,1:11]))[, "sil_width"])
}
plot(2:10, sil_width[2:10], type = "b", xlab = "Number of Clusters", ylab = "Silhouette Width")

# Part c: K-means clustering investigation
# Choosing the most favored k
best_k <- 3  # Setting the number of clusters before preprocessing to 3
kmeans_model <- kmeans(wine_data_no_outliers[,1:11], centers = best_k)

# Displaying k-means output
print(kmeans_model)

# Cluster plot
fviz_cluster(kmeans_model, data = wine_data_no_outliers[,1:11], geom = "point", stand = FALSE)

# Calculate BSS and WSS indices
BSS <- sum(kmeans_model$betweenss)
TSS <- sum(kmeans_model$tot.withinss + kmeans_model$betweenss)
WSS <- sum(kmeans_model$tot.withinss)
cat("BSS/TSS ratio:", BSS/TSS, "\n")
cat("BSS:", BSS, "\n")
cat("WSS:", WSS, "\n")

# Part d: Silhouette plot
sil_plot <- silhouette(kmeans_model$cluster, dist(wine_data_no_outliers[,1:11]))
mean_sil_width <- mean(sil_plot[, "sil_width"])
plot(sil_plot, main = "Silhouette Plot")
abline(h = mean_sil_width, col = "red")
cat("Average silhouette width:", mean_sil_width, "\n")


# Part e: PCA analysis
pca_result <- prcomp(wine_data_no_outliers[,1:11], scale. = TRUE)
summary(pca_result)

# Scree plot
fviz_eig(pca_result, addlabels = TRUE)

# Eigenvalues of principal components
eigen_values <- pca_result$sdev^2
eigen_values

# Cumulative variance explained
cumulative_variance <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
cumulative_variance

# Part f: Finding appropriate k for new k-means clustering after PCA
# Selecting PCs with cumulative variance > 85%
selected_pcs <- which(cumulative_variance > 0.85)[1]

# Creating transformed dataset with selected PCs
transformed_data <- as.data.frame(predict(pca_result, newdata = wine_data_no_outliers[,1:11])[, 1:selected_pcs])

# NBclust for transformed dataset
nb_results_pca <- NbClust(transformed_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "complete", index = "all")
nb_results_pca$Best.nc[1]  # Best number of clusters suggested by NBclust

# Gap statistics for transformed dataset
set.seed(123)
gap_stat_pca <- clusGap(transformed_data, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
plot(gap_stat_pca, main = "Gap Statistic Plot - PCA")

# Part g: K-means clustering using transformed dataset
best_k_pca <- 2  # Setting the number of clusters after PCA to 2
kmeans_model_pca <- kmeans(transformed_data, centers = best_k_pca)

# Displaying k-means output
print(kmeans_model_pca)

# Cluster plot
fviz_cluster(kmeans_model_pca, data = transformed_data, geom = "point", stand = FALSE)

# Calculate BSS and WSS indices
BSS_pca <- sum(kmeans_model_pca$betweenss)
TSS_pca <- sum(kmeans_model_pca$tot.withinss + kmeans_model_pca$betweenss)
WSS_pca <- sum(kmeans_model_pca$tot.withinss)
cat("BSS/TSS ratio (PCA):", BSS_pca/TSS_pca, "\n")
cat("BSS (PCA):", BSS_pca, "\n")
cat("WSS (PCA):", WSS_pca, "\n")

# Part h: Silhouette plot for transformed dataset
sil_plot_pca <- silhouette(kmeans_model_pca$cluster, dist(transformed_data))
mean_sil_width_pca <- mean(sil_plot_pca[, "sil_width"])
plot(sil_plot_pca, main = "Silhouette Plot - PCA")
abline(h = mean_sil_width_pca, col = "red")
cat("Average silhouette width (PCA):", mean_sil_width_pca, "\n")


# Load required library
library(fpc)

# Initialize empty vectors to store the Calinski-Harabasz index and number of clusters
calinski_indices <- c()
num_clusters <- c()

# Iterate over different numbers of clusters
for (k in 2:10) {
  # Perform K-means clustering with k clusters
  kmeans_model <- kmeans(transformed_data, centers = k)
  
  # Convert transformed data to distance matrix
  dist_matrix <- dist(transformed_data)
  
  # Calculate Calinski-Harabasz index
  calinski_index <- cluster.stats(dist_matrix, kmeans_model$cluster)$ch
  
  # Append results to vectors
  calinski_indices <- c(calinski_indices, calinski_index)
  num_clusters <- c(num_clusters, k)
}

# Plot Calinski-Harabasz index against number of clusters
plot(num_clusters, calinski_indices, type = "b", xlab = "Number of Clusters", ylab = "Calinski-Harabasz Index", main = "Calinski-Harabasz Index vs. Number of Clusters")
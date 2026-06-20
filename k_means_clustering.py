import torch

def k_means_clustering(points, k, initial_centroids, max_iterations) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float)
    # Your implementation here
    for _ in range(max_iterations):

        # step 1
        distances = torch.cdist(points_t, centroids)

        labels = torch.argmin(distances, dim = 1)
        
        new_centroids = []

        for i in range(k):
            new_clusters = points_t[labels==i] 

            if len(new_clusters) > 0:
                new_centroids.append(new_clusters.mean(dim = 0))
            else:
                new_centroids.append(centroids[i])
        new_centroids = torch.stack(new_centroids)

        if torch.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return [
            tuple(round(float(x), 4) for x in centroid)
            for centroid in centroids
            ]
    pass


print(k_means_clustering(points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)],
k = 2,
initial_centroids = [(1, 1), (10, 1)],
max_iterations = 10))





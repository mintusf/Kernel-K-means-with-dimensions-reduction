from library import definitions as model


# The following program applies k-means algorithm for unsupervised learning 
# using sklearn circles dataset. To create a decision boundary in the case of
# a non-linear dataset, Gaussian kernels are utilized. In order to decrease
# computational heaviness, dimensions reduction using principal component analysis 
# was applied. All functions were written from scratch without using any built-in
# sklearn functions.

   

def run(trials=10, var_limit=0.999, tested_sigma=[1]):
    X_data = model.init()
    (centroids, assigned_centroids) = model.train(X_data, trials, var_limit, tested_sigma)
 
if __name__ == '__main__':
    run()

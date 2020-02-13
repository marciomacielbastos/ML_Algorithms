class KMeansDistanceDistributions:
    def __init__(self, n_clusters=2, tol=1e-4, max_iter=300, random_state=0):
        self.seed = random_state
        np.random.seed(seed=random_state)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.L = None

    def start_cluster_centers(self, X):
        # Initiate the points equally spaced around the n-sphere centered on the mean       
        min_ = tf.reduce_min(X, axis=0)
        max_ = tf.reduce_max(X, axis=0)
        r = tf.reduce_min(max_ - min_, name='radius')/2
        O = tfp.stats.percentile(X, q=50, axis=0)

        ϕ = [(2 * np.pi / self.n_clusters) * np.array(range(self.n_clusters))]
        ϕ.append((np.pi / self.n_clusters) * np.array(range(self.n_clusters)))
        ϕ = tf.Variable(ϕ, dtype=tf.float32, name=r'$\phi$')

        sin = tf.math.sin(ϕ)
        cos = tf.math.cos(ϕ)

        D = X.shape[1]
        coord = []
        for i in range(self.n_clusters):
            x = [r * cos[0][i]]
            for j in range(1, D - 1):
                x.append(r * tf.math.pow(sin[1][i], j - 1) * sin[0][i] * cos[1][i])
            x.append(r * tf.math.pow(sin[1][i], D - 1) * sin[0][i])
            coord.append(x)
        self.cluster_centers_ = tf.Variable(tf.stack(coord + O), name='clusters O')
    
    def fit(self,X):
        self.start_cluster_centers(X)
        
        points_expanded = tf.expand_dims(X, 0)
        centroids_expanded = tf.expand_dims(self.cluster_centers_,1)
        distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
        self.labels_ = tf.argmin(distances, 0)
        
        for i in range(self.max_iter):            
            means = []
            for c in range(self.n_clusters):
                g = tf.gather(X, tf.reshape(tf.where(tf.equal(self.labels_, c)),[1,-1]))
                if tf.size(g) > 0:
                    m = tf.reduce_mean(g, axis=1)
                    means.append(m)
                else:
                    means.append(tf.Variable(np.random.uniform(
                                tf.reduce_min(X, axis=0).numpy().tolist(), 
                                tf.reduce_min(X, axis=0).numpy().tolist(), (1,X.shape[1])), dtype=tf.float32))
            new_centroids = tf.concat(means, 0) 
            val = (tf.norm(new_centroids - self.cluster_centers_)).numpy()
            if val < 1e-4:
                centroids_expanded = tf.expand_dims(tf.concat(means, 0), 1)
                distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
                new_centroids = tf.gather(X,tf.argmin(distances, 1))
                self.cluster_centers_.assign(new_centroids)
                break
            self.cluster_centers_.assign(new_centroids)
            centroids_expanded = tf.expand_dims(self.cluster_centers_, 1)
            distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
            self.labels_ = tf.argmin(distances, 0)
        return self
    
    def MHD(self, X):
        self.mean = []
        self.L = []
        MHD_ = []
        for c in range(self.n_clusters):
            # L calc
            g = tf.gather(X, tf.reshape(tf.where(tf.equal(self.labels_, c)),[1,-1]))           

            mean_expanded = tf.expand_dims(self.cluster_centers_[c], -1)
            
            cov = tfp.stats.covariance(g[0])
            l = tf.linalg.cholesky(cov)
            
            # MHD calc
            delta = (tf.transpose(pts_g) - mean_expanded)
            z = tf.linalg.triangular_solve(self.L[i], delta)
            mhd = tf.reduce_min(tf.reduce_sum(tf.square(z),1), 0)
            
            self.mean.append(m)
            self.L.append(l)
            MHD_.append(mhd)
            
        self.L = tf.Variable(self.L)
        return tf.Variable(MHD_)
    
    def fit_mhd(self, X):
        self.fit(X)
        mhd = self.MHD(X)
        return mhd

    def predict(self, X):
        data_expanded = tf.expand_dims(X, 0)
        centroids_expanded = tf.expand_dims(self.cluster_centers_, 1)
        distances = tf.reduce_sum(tf.square(tf.subtract(data_expanded, centroids_expanded)), 2)
        assignments = tf.argmin(distances, 0)
        return assignments.numpy()
    
    def dump(self, X):
        data = {}
        data['MHD'] = self.fit_mhd(X)
        data['L'] = self.L
        data['cluster_centers_'] = self.cluster_centers_
        return json.dumps(data)
    
    def set_params(self, cc):
        self.cluster_centers_ = cc

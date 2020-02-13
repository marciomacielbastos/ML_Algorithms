class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):        
        self.eps2 = eps*eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def get_neighbors(self, X, pto):
        pto_exp = tf.expand_dims(X[pto], 0)
        pts_exp = tf.expand_dims(X, 0)
        distances = tf.reduce_sum(tf.square(tf.subtract(pto_exp, pts_exp)), 2)    
        neighbors_idx = tf.unique(tf.reshape(tf.where(tf.less_equal(distances, self.eps2)),[1,-1])[0])[0]
        return neighbors_idx

    def assign_cluster(self, X, pto, C):
        queue = [pto]
        i = 0
        while i < len(queue):           
            pto = queue[i]
            neighbors = self.get_neighbors(X, pto)
            if neighbors.shape[0] < self.min_samples:
                i += 1
                continue            
            for n in neighbors:
                if (self.labels_[n] == -1):
                    self.labels_[n] = C  
                elif (self.labels_[n] == 0):
                    self.labels_[n] = C
                    queue.append(n)
            i += 1 
    
    def fit(self, X):
        self.labels_ = np.zeros(X.shape[0], dtype=int)   
        C = 0
        for pto in range(X.shape[0]):
            if (self.labels_[pto] != 0):
                continue
            neighbors = self.get_neighbors(X, pto)
            if (neighbors.shape[0] < self.min_samples):
                self.labels_[pto] = -1
            else: 
                C += 1
                self.labels_[pto] = C
                self.assign_cluster(X, pto, C)
        return self

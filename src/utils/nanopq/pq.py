import numpy as np
from scipy.cluster.vq import kmeans2, vq
from scipy import stats
import hdbscan
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sqlalchemy import true
from torch_scatter import scatter_mean
import torch

class PQ(object):
    """Pure python implementation of Product Quantization (PQ) [Jegou11]_.

    For the indexing phase of database vectors,
    a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `Ks` codewords.
    For the querying phase, given a new `D`-dim query vector, the distance beween the query
    and the database PQ-codes are efficiently approximated via Asymmetric Distance.

    All vectors must be np.ndarray with np.float32

    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011

    Args:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        verbose (bool): Verbose flag

    Attributes:
        M (int): The number of sub-space
        Ks (int): The number of codewords for each subspace
        verbose (bool): Verbose flag
        code_dtype (object): dtype of PQ-code. Either np.uint{8, 16, 32}
        codewords (np.ndarray): shape=(M, Ks, Ds) with dtype=np.float32.
            codewords[m][ks] means ks-th codeword (Ds-dim) for m-th subspace
        Ds (int): The dim of each sub-vector, i.e., Ds=D/M

    """

    def __init__(self, M, Ks=256, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = (
            np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        )
        self.codewords = None
        self.labels = None
        self.Ds = None

        if verbose:
            print("M: {}, Ks: {}, code_dtype: {}".format(M, Ks, self.code_dtype))

    def __eq__(self, other):
        if isinstance(other, PQ):
            return (self.M, self.Ks, self.verbose, self.code_dtype, self.Ds) == (
                other.M,
                other.Ks,
                other.verbose,
                other.code_dtype,
                other.Ds,
            ) and np.array_equal(self.codewords, other.codewords)
        else:
            return False

    def fit(self, vecs, iter=20, seed=123):
        """Given training vectors, run k-means for each sub-space and create
        codewords for each sub-space.

        This function should be run once first of all.

        Args:
            vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            iter (int): The number of iteration for k-means
            seed (int): The seed for random process

        Returns:
            object: self

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert D % self.M == 0, "input dimension must be dividable by M"
        self.Ds = int(D / self.M)
        self.revec = None

        np.random.seed(seed)
        if self.verbose:
            print("iter: {}, seed: {}".format(iter, seed))

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, self.Ds), dtype=np.float32)
        self.labels = np.zeros((self.M, N), dtype=np.float32)
        self.revec = np.zeros_like(vecs)
        for m in range(self.M):
            if self.verbose:
                print("Training the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
            # self.codewords[m], a = kmeans2(vecs_sub, self.Ks, iter=iter, minit="points")
            # KMeans
            self.codewords[m], self.labels[m] = kmeans2(vecs_sub, self.Ks, iter=iter, minit="points")
            
            # HDBSCAN
            # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
            # cluster_labels = clusterer.fit_predict(vecs_sub)
            # self.labels[m] = cluster_labels
            # src = torch.Tensor(vecs_sub[cluster_labels>0])
            # index = torch.from_numpy(cluster_labels[cluster_labels>0])
            # out = scatter_mean(src, index, dim=0)
            # self.codewords[m] = out

            # Spectral Clustering
            # clusterer = SpectralClustering(n_clusters=self.Ks, assign_labels='discretize', random_state=0)
            # cluster_labels = clusterer.fit_predict(vecs_sub)
            # self.labels[m] = cluster_labels
            # src = torch.Tensor(vecs_sub[cluster_labels>0])
            # index = torch.from_numpy(cluster_labels[cluster_labels>0])
            # out = scatter_mean(src, index, dim=0)
            # self.codewords[m] = out

            # GMM
            # clusterer = GaussianMixture(n_components=self.Ks, random_state=0)
            # cluster_labels = clusterer.fit_predict(vecs_sub)
            # self.labels[m] = cluster_labels
            # src = torch.Tensor(vecs_sub[cluster_labels>0])
            # index = torch.from_numpy(cluster_labels[cluster_labels>0])
            # out = scatter_mean(src, index, dim=0)
            # self.codewords[m] = out
            
        
        for m in range(self.M):
            for n in range(N):
                self.revec[n][m * self.Ds: (m + 1) * self.Ds] = self.codewords[m][int(self.labels[m][n])]

        return self.revec

    def encode(self, vecs):
        """Encode input vectors into PQ-codes.

        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.

        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert D == self.Ds * self.M, "input dimension must be Ds * M"

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            if self.verbose:
                print("Encoding the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
            codes[:, m], _ = vq(vecs_sub, self.codewords[m])

        return codes

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors
        approximately by fetching the codewords.

        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code

        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32

        """
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.Ds * self.M), dtype=np.float32)
        for m in range(self.M):
            vecs[:, m * self.Ds : (m + 1) * self.Ds] = self.codewords[m][codes[:, m], :]

        return vecs

    def dtable(self, query):
        """Compute a distance table for a query vector.
        The distances are computed by comparing each sub-vector of the query
        to the codewords for each sub-subspace.
        `dtable[m][ks]` contains the squared Euclidean distance between
        the `m`-th sub-vector of the query and the `ks`-th codeword
        for the `m`-th sub-space (`self.codewords[m][ks]`).

        Args:
            query (np.ndarray): Input vector with shape=(D, ) and dtype=np.float32

        Returns:
            nanopq.DistanceTable:
                Distance table. which contains
                dtable with shape=(M, Ks) and dtype=np.float32

        """
        assert query.dtype == np.float32
        assert query.ndim == 1, "input must be a single vector"
        (D,) = query.shape
        assert D == self.Ds * self.M, "input dimension must be Ds * M"

        # dtable[m] : distance between m-th subvec and m-th codewords (m-th subspace)
        # dtable[m][ks] : distance between m-th subvec and ks-th codeword of m-th codewords
        dtable = np.empty((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            query_sub = query[m * self.Ds : (m + 1) * self.Ds]
            dtable[m, :] = np.linalg.norm(self.codewords[m] - query_sub, axis=1) ** 2

        return DistanceTable(dtable)


class DistanceTable(object):
    """Distance table from query to codeworkds.
    Given a query vector, a PQ/OPQ instance compute this DistanceTable class
    using :func:`PQ.dtable` or :func:`OPQ.dtable`.
    The Asymmetric Distance from query to each database codes can be computed
    by :func:`DistanceTable.adist`.

    Args:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32
            computed by :func:`PQ.dtable` or :func:`OPQ.dtable`

    Attributes:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32.
            Note that dtable[m][ks] contains the squared Euclidean distance between
            (1) m-th sub-vector of query and (2) ks-th codeword for m-th subspace.

    """

    def __init__(self, dtable):
        assert dtable.ndim == 2
        assert dtable.dtype == np.float32
        self.dtable = dtable

    def adist(self, codes):
        """Given PQ-codes, compute Asymmetric Distances between the query (self.dtable)
        and the PQ-codes.

        Args:
            codes (np.ndarray): PQ codes with shape=(N, M) and
                dtype=pq.code_dtype where pq is a pq instance that creates the codes

        Returns:
            np.ndarray: Asymmetric Distances with shape=(N, ) and dtype=np.float32

        """

        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.dtable.shape[0]

        # Fetch distance values using codes. The following codes are
        dists = np.sum(self.dtable[range(M), codes], axis=1)

        # The above line is equivalent to the followings:
        # dists = np.zeros((N, )).astype(np.float32)
        # for n in range(N):
        #     for m in range(M):
        #         dists[n] += self.dtable[m][codes[n][m]]

        return dists

class CPQ(object):

    def __init__(self, M, Ks=256, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = (
            np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        )
        self.codewords = None
        self.labels = None
        self.Ds = None

        if verbose:
            print("M: {}, Ks: {}, code_dtype: {}".format(M, Ks, self.code_dtype))

    def __eq__(self, other):
        if isinstance(other, PQ):
            return (self.M, self.Ks, self.verbose, self.code_dtype, self.Ds) == (
                other.M,
                other.Ks,
                other.verbose,
                other.code_dtype,
                other.Ds,
            ) and np.array_equal(self.codewords, other.codewords)
        else:
            return False

    def fit(self, vecs, iter=20, seed=123):
        """Given training vectors, run k-means for each sub-space and create
        codewords for each sub-space.

        This function should be run once first of all.

        Args:
            vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            iter (int): The number of iteration for k-means
            seed (int): The seed for random process

        Returns:
            object: self

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, self.D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert self.D > self.M
        self.revec = None

        np.random.seed(seed)
        if self.verbose:
            print("iter: {}, seed: {}".format(iter, seed))

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        # ccenters, self.clabels = kmeans2(vecs.T, self.M, iter=iter, minit="points")
        clustering = SpectralClustering(self.M, affinity='precomputed', n_init=100, assign_labels='discretize')
        # clustering = AgglomerativeClustering(n_clusters=self.M, affinity='precomputed', linkage='single')
        rho, _ = stats.spearmanr(vecs)
        # rho = np.corrcoef(vecs.T)
        dist = 1 - abs(rho)
        # dist = 1/np.exp(dist)
        self.clabels = clustering.fit_predict(dist)

        _, self.Ds = np.unique(self.clabels, return_counts=True)
        self.codewords = [np.zeros((self.Ks, d), dtype=np.float32) for d in self.Ds]
        self.labels = np.zeros((self.M, N), dtype=np.float32)
        self.revec = np.zeros_like(vecs)
        for m in range(self.M):
            if self.verbose:
                print("Training the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, self.clabels == m]
            # self.codewords[m], a = kmeans2(vecs_sub, self.Ks, iter=iter, minit="points")
            # KMeans
            self.codewords[m], self.labels[m] = kmeans2(vecs_sub, self.Ks, iter=iter, minit="points")
            
            # HDBSCAN
            # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
            # cluster_labels = clusterer.fit_predict(vecs_sub)
            # self.labels[m] = cluster_labels
            # src = torch.Tensor(vecs_sub[cluster_labels>0])
            # index = torch.from_numpy(cluster_labels[cluster_labels>0])
            # out = scatter_mean(src, index, dim=0)
            # self.codewords[m] = out

            # Spectral Clustering
            # clusterer = SpectralClustering(n_clusters=self.Ks, assign_labels='discretize', random_state=0)
            # cluster_labels = clusterer.fit_predict(vecs_sub)
            # self.labels[m] = cluster_labels
            # src = torch.Tensor(vecs_sub[cluster_labels>0])
            # index = torch.from_numpy(cluster_labels[cluster_labels>0])
            # out = scatter_mean(src, index, dim=0)
            # self.codewords[m] = out

            # GMM
            # clusterer = GaussianMixture(n_components=self.Ks, random_state=0)
            # cluster_labels = clusterer.fit_predict(vecs_sub)
            # self.labels[m] = cluster_labels
            # src = torch.Tensor(vecs_sub[cluster_labels>0])
            # index = torch.from_numpy(cluster_labels[cluster_labels>0])
            # out = scatter_mean(src, index, dim=0)
            # self.codewords[m] = out
            
        
        for m in range(self.M):
            for n in range(N):
                self.revec[n][self.clabels == m] = self.codewords[m][int(self.labels[m][n])]

        return self.revec

    def encode(self, vecs):
        """Encode input vectors into PQ-codes.

        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.

        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert D == self.D, "input dimension must be D"

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            if self.verbose:
                print("Encoding the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, self.clabels == m]
            codes[:, m], _ = vq(vecs_sub, self.codewords[m])

        return codes

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors
        approximately by fetching the codewords.

        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code

        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32

        """
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.D), dtype=np.float32)
        for m in range(self.M):
            vecs[:, self.clabels == m] = self.codewords[m][codes[:, m], :]

        return vecs

    def dtable(self, query):
        """Compute a distance table for a query vector.
        The distances are computed by comparing each sub-vector of the query
        to the codewords for each sub-subspace.
        `dtable[m][ks]` contains the squared Euclidean distance between
        the `m`-th sub-vector of the query and the `ks`-th codeword
        for the `m`-th sub-space (`self.codewords[m][ks]`).

        Args:
            query (np.ndarray): Input vector with shape=(D, ) and dtype=np.float32

        Returns:
            nanopq.DistanceTable:
                Distance table. which contains
                dtable with shape=(M, Ks) and dtype=np.float32

        """
        assert query.dtype == np.float32
        assert query.ndim == 1, "input must be a single vector"
        (D,) = query.shape
        assert D == self.D, "input dimension must be D"

        # dtable[m] : distance between m-th subvec and m-th codewords (m-th subspace)
        # dtable[m][ks] : distance between m-th subvec and ks-th codeword of m-th codewords
        dtable = np.empty((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            query_sub = query[self.clabels == m]
            dtable[m, :] = np.linalg.norm(self.codewords[m] - query_sub, axis=1) ** 2

        return DistanceTable(dtable)


class DistanceTable(object):
    """Distance table from query to codeworkds.
    Given a query vector, a PQ/OPQ instance compute this DistanceTable class
    using :func:`PQ.dtable` or :func:`OPQ.dtable`.
    The Asymmetric Distance from query to each database codes can be computed
    by :func:`DistanceTable.adist`.

    Args:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32
            computed by :func:`PQ.dtable` or :func:`OPQ.dtable`

    Attributes:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32.
            Note that dtable[m][ks] contains the squared Euclidean distance between
            (1) m-th sub-vector of query and (2) ks-th codeword for m-th subspace.

    """

    def __init__(self, dtable):
        assert dtable.ndim == 2
        assert dtable.dtype == np.float32
        self.dtable = dtable

    def adist(self, codes):
        """Given PQ-codes, compute Asymmetric Distances between the query (self.dtable)
        and the PQ-codes.

        Args:
            codes (np.ndarray): PQ codes with shape=(N, M) and
                dtype=pq.code_dtype where pq is a pq instance that creates the codes

        Returns:
            np.ndarray: Asymmetric Distances with shape=(N, ) and dtype=np.float32

        """

        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.dtable.shape[0]

        # Fetch distance values using codes. The following codes are
        dists = np.sum(self.dtable[range(M), codes], axis=1)

        # The above line is equivalent to the followings:
        # dists = np.zeros((N, )).astype(np.float32)
        # for n in range(N):
        #     for m in range(M):
        #         dists[n] += self.dtable[m][codes[n][m]]

        return dists

class RPQ(object):

    def __init__(self, M, Ks=256, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = (
            np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        )
        self.codewords = None
        self.labels = None
        self.Ds = None

        if verbose:
            print("M: {}, Ks: {}, code_dtype: {}".format(M, Ks, self.code_dtype))

    def __eq__(self, other):
        if isinstance(other, PQ):
            return (self.M, self.Ks, self.verbose, self.code_dtype, self.Ds) == (
                other.M,
                other.Ks,
                other.verbose,
                other.code_dtype,
                other.Ds,
            ) and np.array_equal(self.codewords, other.codewords)
        else:
            return False

    def fit(self, vecs, iter=20, seed=123):
        """Given training vectors, run k-means for each sub-space and create
        codewords for each sub-space.

        This function should be run once first of all.

        Args:
            vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            iter (int): The number of iteration for k-means
            seed (int): The seed for random process

        Returns:
            object: self

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, self.D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert self.D > self.M
        self.revec = None
        self.Ds = int(self.D / self.M)

        np.random.seed(seed)
        if self.verbose:
            print("iter: {}, seed: {}".format(iter, seed))

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        # ccenters, self.clabels = kmeans2(vecs.T, self.M, iter=iter, minit="points")
        # clustering = SpectralClustering(self.M, affinity='precomputed', n_init=100, assign_labels='discretize')
        # clustering = AgglomerativeClustering(n_clusters=self.M, affinity='precomputed', linkage='single')
        # rho, _ = stats.spearmanr(vecs)
        # rho = np.corrcoef(vecs.T)
        # rho = 1 - abs(rho)
        # self.clabels = clustering.fit_predict(rho)
        self.clabels = [[i] * self.Ds for i in range(self.M)]
        self.clabels = np.array(self.clabels).flatten()
        shuffle_id = np.random.permutation(np.arange(self.D))
        self.clabels = self.clabels[shuffle_id]

        _, self.Ds = np.unique(self.clabels, return_counts=True)
        self.codewords = [np.zeros((self.Ks, d), dtype=np.float32) for d in self.Ds]
        self.labels = np.zeros((self.M, N), dtype=np.float32)
        self.revec = np.zeros_like(vecs)
        for m in range(self.M):
            if self.verbose:
                print("Training the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, self.clabels == m]
            # self.codewords[m], a = kmeans2(vecs_sub, self.Ks, iter=iter, minit="points")
            # KMeans
            self.codewords[m], self.labels[m] = kmeans2(vecs_sub, self.Ks, iter=iter, minit="points")
            
            # HDBSCAN
            # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
            # cluster_labels = clusterer.fit_predict(vecs_sub)
            # self.labels[m] = cluster_labels
            # src = torch.Tensor(vecs_sub[cluster_labels>0])
            # index = torch.from_numpy(cluster_labels[cluster_labels>0])
            # out = scatter_mean(src, index, dim=0)
            # self.codewords[m] = out

            # Spectral Clustering
            # clusterer = SpectralClustering(n_clusters=self.Ks, assign_labels='discretize', random_state=0)
            # cluster_labels = clusterer.fit_predict(vecs_sub)
            # self.labels[m] = cluster_labels
            # src = torch.Tensor(vecs_sub[cluster_labels>0])
            # index = torch.from_numpy(cluster_labels[cluster_labels>0])
            # out = scatter_mean(src, index, dim=0)
            # self.codewords[m] = out

            # GMM
            # clusterer = GaussianMixture(n_components=self.Ks, random_state=0)
            # cluster_labels = clusterer.fit_predict(vecs_sub)
            # self.labels[m] = cluster_labels
            # src = torch.Tensor(vecs_sub[cluster_labels>0])
            # index = torch.from_numpy(cluster_labels[cluster_labels>0])
            # out = scatter_mean(src, index, dim=0)
            # self.codewords[m] = out
            
        
        for m in range(self.M):
            for n in range(N):
                self.revec[n][self.clabels == m] = self.codewords[m][int(self.labels[m][n])]

        return self.revec

    def encode(self, vecs):
        """Encode input vectors into PQ-codes.

        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.

        Returns:
            np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype

        """
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert D == self.D, "input dimension must be D"

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            if self.verbose:
                print("Encoding the subspace: {} / {}".format(m, self.M))
            vecs_sub = vecs[:, self.clabels == m]
            codes[:, m], _ = vq(vecs_sub, self.codewords[m])

        return codes

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors
        approximately by fetching the codewords.

        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code

        Returns:
            np.ndarray: Reconstructed vectors with shape=(N, D) and dtype=np.float32

        """
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.D), dtype=np.float32)
        for m in range(self.M):
            vecs[:, self.clabels == m] = self.codewords[m][codes[:, m], :]

        return vecs

    def dtable(self, query):
        """Compute a distance table for a query vector.
        The distances are computed by comparing each sub-vector of the query
        to the codewords for each sub-subspace.
        `dtable[m][ks]` contains the squared Euclidean distance between
        the `m`-th sub-vector of the query and the `ks`-th codeword
        for the `m`-th sub-space (`self.codewords[m][ks]`).

        Args:
            query (np.ndarray): Input vector with shape=(D, ) and dtype=np.float32

        Returns:
            nanopq.DistanceTable:
                Distance table. which contains
                dtable with shape=(M, Ks) and dtype=np.float32

        """
        assert query.dtype == np.float32
        assert query.ndim == 1, "input must be a single vector"
        (D,) = query.shape
        assert D == self.D, "input dimension must be D"

        # dtable[m] : distance between m-th subvec and m-th codewords (m-th subspace)
        # dtable[m][ks] : distance between m-th subvec and ks-th codeword of m-th codewords
        dtable = np.empty((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            query_sub = query[self.clabels == m]
            dtable[m, :] = np.linalg.norm(self.codewords[m] - query_sub, axis=1) ** 2

        return DistanceTable(dtable)


class DistanceTable(object):
    """Distance table from query to codeworkds.
    Given a query vector, a PQ/OPQ instance compute this DistanceTable class
    using :func:`PQ.dtable` or :func:`OPQ.dtable`.
    The Asymmetric Distance from query to each database codes can be computed
    by :func:`DistanceTable.adist`.

    Args:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32
            computed by :func:`PQ.dtable` or :func:`OPQ.dtable`

    Attributes:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32.
            Note that dtable[m][ks] contains the squared Euclidean distance between
            (1) m-th sub-vector of query and (2) ks-th codeword for m-th subspace.

    """

    def __init__(self, dtable):
        assert dtable.ndim == 2
        assert dtable.dtype == np.float32
        self.dtable = dtable

    def adist(self, codes):
        """Given PQ-codes, compute Asymmetric Distances between the query (self.dtable)
        and the PQ-codes.

        Args:
            codes (np.ndarray): PQ codes with shape=(N, M) and
                dtype=pq.code_dtype where pq is a pq instance that creates the codes

        Returns:
            np.ndarray: Asymmetric Distances with shape=(N, ) and dtype=np.float32

        """

        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.dtable.shape[0]

        # Fetch distance values using codes. The following codes are
        dists = np.sum(self.dtable[range(M), codes], axis=1)

        # The above line is equivalent to the followings:
        # dists = np.zeros((N, )).astype(np.float32)
        # for n in range(N):
        #     for m in range(M):
        #         dists[n] += self.dtable[m][codes[n][m]]

        return dists
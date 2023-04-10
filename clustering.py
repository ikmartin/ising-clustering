from abc import ABC, abstractmethod


class Cluster:
    """
    A class representing a single cluster in a clustering

    ...

    Attributes
    ----------
    indices : list[int]
        the indices of the data in this cluster.
    id_num : int
        the id_number of this cluster. Set to -1 by default.
    self.size : int
        the number of data points in this cluster.

    Methods
    -------
    info() -> str
        returns string containing basic info about this cluster
    to_json() -> str
        generate json string representing cluster (NOT IMPLEMENTED)


    Static Methods
    --------------
    from_json(file: str) -> Cluster
        generates Cluster from a json string (NOT IMPLEMENTED)
    """

    def __init__(self, indices: list[int], id_num: int = -1, center: int = -1):
        """
        Parameters
        ----------
        indices : list[int]
            the indices of the data in this cluster
        id_num : int
            the id_number of this cluster. Set to -1 by default.

        """
        self.id_num = id_num
        self.indices = indices
        self.size = len(indices)
        self.center = center

    def info(self):
        """Gerenate an info string for this cluster

        Returns
        -------
        str
            a string describing class attributes separated by \n
        """
        return f"size: {self.size}"

    def to_json(self):
        """Generates json for this cluster

        Returns
        -------
        str
            a string containing the json for serializing this cluster
        """
        pass

    @staticmethod
    def from_json(json: str):
        """Returns an instance of cluster generated from json

        Parameters
        ----------
        json: str
            json string from which to generate class

        Returns
        -------
        Cluster
            a single Cluster
        """
        pass


class RefinedCluster(Cluster):
    """
    A class representing a single cluster representing part of a refined clustering structure. Points to the parent cluster.

    ...

    Attributes
    ----------
    data : list
        the data contained in cluster
    size : int
        the number of elements in this cluster
    kind : str
        a description of the cluster type
    parent : RefinedCluster
        a pointer to the cluster from which this is refined

    """

    def __init__(self, indices, id_num: int, parent: Cluster):
        super().__init__(id_num=id_num, indices=indices)
        self.parent = parent
        self.parent_id = self.parent.id_num if parent != None else None


class Clustering:
    """
    A clustering of data

    ...

    Attributes
    ----------
    clusters : list(Type: Cluster)
        the partition into clusters
    self.size : int
        the number of clusters

    Methods
    -------

    """

    def __init__(self, clusters: list[Cluster]):
        self.clusters = clusters
        self.size = len(clusters)
        self._current_index = 0

    def __getitem__(self, key):
        return self.clusters[key]

    def __iter__(self):
        """Makes this object iterable"""
        return self

    def __next__(self):
        """Returns the next spin in the iteration formatted in the appropriate mode. Converts _current_index to the appropriate spin."""
        if self._current_index >= self.size:
            # reset the current index to 0
            self._current_index = 0
            raise StopIteration

        index = self._current_index
        self._current_index += 1

        return self.clusters[index]


class Model:
    """
    An implementation of a clustering model

    ...

    Attributes
    ----------
    data : Uknown (must be iterable)
        the data to be clustered
    size : int
        the number of data points
    clusters : list(Type Cluster)
        the partition into clusters

    Methods
    -------
    model(self) -> Clustering
        generates the clustering
    """

    def __init__(self, data, size):
        self.data = data
        self.size = size

    def model(self):
        raise NotImplementedError


class RefinedClustering(Model):
    """
    Clustering model which refines clusters one at a time

    ...

    Attributes
    ----------
    data :

    Methods
    -------
    verify() -> str
        verify that this is a valid clustering and return status

    """

    def __init__(self, data, size: int):
        """
        The initializer for this class.

        Arguments
        ---------
        data : Uknown
            the data to be modeled
        refine_criterion : method(Clustering) -> bool
            determines if the passed clustering (Clustering) needs to be refined

        """
        super().__init__(data=data, size=size)

    @abstractmethod
    def refine_criterion(self, cluster: Cluster) -> bool:
        pass

    @abstractmethod
    def refine(
        self, cluster: Cluster, current_id: int
    ) -> tuple[RefinedCluster, RefinedCluster]:
        pass

    def model(self):
        """
        Performs the refined clustering.
        """
        # set up the clustering
        indices = list(range(self.size))  # indices of all data points
        print(indices)
        basecluster = Clustering([Cluster(id_num=0, indices=indices)])
        clusterings = [basecluster]
        current_count = 1

        done = False
        while done == False:
            done = True
            # iterate through the most recent clustering
            newlayer = []
            for cluster in clusterings[-1]:
                # if refine_criterion is met, then refine the cluster
                if self.refine_criterion(cluster):
                    # if this is the first refinement thus far, set 'done' to False
                    if done == True:
                        done = False

                    # refine the cluster
                    clust1, clust2 = self.refine(cluster, current_count)
                    newlayer += [clust1, clust2]
                    current_count += 2

                # if we don't need to refine, preserve cluster
                else:
                    newlayer.append(cluster)
                    current_count += 1

            if done == False:
                clusterings.append(Clustering(clusters=newlayer))

        return clusterings

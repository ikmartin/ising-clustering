class Cluster:
    """
    A class representing a single cluster in a clustering

    ...

    Attributes
    ----------
    data : list
        the data contained in cluster
    size : int
        the number of elements in this cluster
    kind : str
        a description of the cluster type

    Methods
    -------
    info() -> str
        returns string containing basic info about this cluster
    to_json() -> str
        generate json string representing cluster


    Static Methods
    --------------
    from_json(file: str) -> Cluster
        generates Cluster from a json string
    """

    def __init__(self, id_num: int, indices):
        """
        Parameters
        ----------
        data : list
            The data comprising the cluster
        """
        self.id_num = id_num
        self.indices = indices
        self.size = len(indices)

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

    def __init__(self, id_num: int, indices, parent=None):
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

    Methods
    -------
    verify() -> str
        verify that this is a valid clustering and return status

    """

    def __init__(self, clusters):
        self.clusters = clusters
        self.k = len(clusters)


class RefinedClustering(Clustering):
    """
    A choice of clustering refined from some other clustering. Data structure, implementation handled elsewhere

    ...

    Attributes
    ----------
    cluster : list(RefinedCluster)
        the partition of a data set into clusters

    Methods
    -------
    verify() -> str
        verify that this is a valid clustering and return status

    """

    def __init__(self, k: int, generation: int):
        self.k = k
        self.generation = generation


class Model:
    """
    An implementation of a clustering model

    ...

    Attributes
    ----------
    data : list
        the data to be clustered
    clusters : list(Type Cluster)
        the partition into clusters

    Methods
    -------
    model(self) -> Clustering
        generates the clustering
    """

    def __init__(self, data):
        self.data = data

    def model(
        self,
    ):
        raise NotImplementedError

from abc import ABC, abstractmethod
from typing import Callable, Optional


class Cluster:
    """
    A class representing a single cluster in a clustering.

    Attributes
    ----------
    indices : list[int]
        the indices of the data in this cluster.
    id_num : int
        the id_number of this cluster. Is the hash of the indices.
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

    def __init__(self, indices: list[int]):
        """
        Parameters
        ----------
        indices : list[int]
            the indices of the data in this cluster
        id_num : int
            the id_number of this cluster. Set to -1 by default.

        """
        self.id_num = hash(tuple(indices))
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
    is_leaf : bool
        indicates whether this is

    """

    def __init__(self, indices, generation: int, parent: Optional[Cluster] = None):
        """Initializes a refined cluster
        Parameters
        ----------
        indices : list[int]
            the indices of the data points in this cluster
        """

        super().__init__(indices=indices)
        self.parent = parent
        self.generation = generation
        self.top = True if parent == None else False


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
        """Initilizes a model.

        Notes
        -----
        - create a model by inheriting from this class and implementing the listed methods.
        - size is not computed from data as data may be not be an array-like data structure.
        """
        self.data = data
        self.size = size

    def model(self):
        raise NotImplementedError


class RefinementClustering(Model):
    """
    Clustering model which refines clusters one at a time

    ...

    Attributes
    ----------
    data
        the data to be modeled
    size : int
        the number of data points
    clustering : list[RefinedCluster]
        the final clustering achieved by the model
    generations : list[list[RefinedCluster]]
        a record of the generations in the refinement

    Methods
    -------
    refine(cluster: Cluster) -> list[RefinedCluster]:
        refines the provided cluster
    model() -> NoReturn:
        performs the refinement clustering

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
        self.generations = []
        self.clusters = []

    @property
    def gen_num(self) -> int:
        return len(self.generations)

    @abstractmethod
    def refine_criterion(self, cluster: RefinedCluster) -> bool:
        pass

    @abstractmethod
    def refine(self, cluster: RefinedCluster) -> list[RefinedCluster]:
        pass

    def initialize(self):
        indices = list(range(self.size))  # indices of all data points
        print(indices)
        self.generations = [
            [RefinedCluster(indices=indices, generation=0, parent=None)]
        ]

    def model(self):
        """
        Performs the refined clustering.
        """
        # set up the clustering
        self.initialize()

        done = False
        while done == False:
            done = True
            # treat the most recent generation
            for cluster in self.generations[-1]:
                condition = self.refine_criterion(cluster)
                # if refinement needed
                if condition:
                    # if this is true, then this is the first refinement of the generation
                    # hence need to add a new generation
                    if done == True:
                        done = False
                        self.generations.append([])

                    self.generations[-1] += self.refine(cluster)

                # if refinement not needed
                else:
                    self.clusters.append(cluster)

        return self.clusters

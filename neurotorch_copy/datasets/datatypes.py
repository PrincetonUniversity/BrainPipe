from numbers import Number


class Vector:
    """
    A basic vector data type
    """
    def __init__(self, *components: Number):
        """
        Initializes a vector
        :param components: Numbers specifying the components of the vector
        :type components: List of numbers
        """
        self.setComponents(components)

    def setComponents(self, components: list):
        """
        Set the components in a vector
        :param components: A list of numbers specifying the vector's 
components
        :type components: List of numbers
        """
        if not all(isinstance(x, Number) for x in components):
            raise ValueError("components must contain all numbers instead" +
                             " it contains {}".format(components))
        self.components = tuple(components)

    def getComponents(self) -> list:
        """
        Retrieves the components of a vector
        :return: A list of numbers specifying the vector's components
        :rtype: List of numbers
        """
        return self.components

    def getDimension(self) -> int:
        """
        Returns the dimension of the vector
        :return: The vector's dimension
        :rtype: int
        """
        return len(self.getComponents())

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise IndexError("the index must be an positive integer")
        if idx < 0 or idx >= self.getDimension():
            raise IndexError("the index is out-of-bounds")

        return self.getComponents()[idx]

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("other must be a vector instead"
                             " it is ".format(type(other)))

        if self.getDimension() != other.getDimension():
            raise ValueError("other must have the same dimension instead "
                             + "self is {} and other is {}".format(self,
                                                                   other))

        result = [s + o for s, o in zip(self.getComponents(),
                                        other.getComponents())]

        return Vector(*result)

    def __mul__(self, other):
        if isinstance(other, Number):
            result = [s * other for s in self.getComponents()]

        elif isinstance(other, Vector):
            result = [s*o for s, o in zip(self.getComponents(),
                                          other.getComponents())]

        else:
            raise ValueError("other must be a number or a vector instead"
                             " it is {}".format(type(other)))

        return Vector(*result)

    def __div__(self, other):
        if isinstance(other, Number):
            return self*(1/other)

        if isinstance(other, Vector):
            result = Vector([1/o for o in other.getComponents()])
            result *= self

            return result

    def __sub__(self, other):
        return self+(other*-1)

    def __neg__(self):
        return self*-1

    def __eq__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("other must be a vector instead"
                             " it is ".format(type(other)))

        if self.getDimension() != other.getDimension():
            raise ValueError("other must have the same dimension")

        return all(r1 == r2 for r1, r2 in zip(self.getComponents(),
                                              other.getComponents()))

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "{}".format(self.getComponents())

    def __iter__(self):
        return iter(self.getComponents())


class BoundingBox:
    """
    A basic data type specifying a cube
    """
    def __init__(self, edge1: Vector, edge2: Vector):
        """
        Initializes a bounding box
        :param edge1: A vector specifying the first edge of the box
        :type edge1: Vector
        :param edge2: A vector specifying the second edge of the box
        :type edge2: Vector
        """
        self.setEdges(edge1, edge2)

    def setEdges(self, edge1: Vector, edge2: Vector):
        """
        Sets the edges of the bounding box
        :param edge1: A vector specifying the first edge of the box
        :type edge1: Vector
        :param edge2: A vector specifying the second edge of the box
        :type edge2: Vector
        """
        if not isinstance(edge1, Vector) or not isinstance(edge2, Vector):
            raise ValueError("edges must be vectors")

        if edge1.getDimension() != edge2.getDimension():
            raise ValueError("edges must have the same dimension instead"
                             + " edge 1 is {} and edge 2 is {}".format(edge1,
                                                                       edge2))

        self.edge1 = edge1
        self.edge2 = edge2

    def getEdges(self) -> tuple:
        """
        Returns the two edges for the bounding box
        :return: A tuple of two vectors containing the edges of the box
        :rtype: tuple
        """
        return (self.edge1, self.edge2)

    def getDimension(self) -> int:
        """
        Returns the dimension of the bounding box
        :return: The dimension of the bounding box
        :rtype: int
        """
        return self.getEdges()[0].getDimension()

    def getSize(self) -> Vector:
        """
        Returns the size of the bounding box as a vector
        :return: The size of the bounding box
        :rtype: Vector
        """
        edge1, edge2 = self.getEdges()
        chunk_size = edge2 - edge1

        return chunk_size

    def getNumpyDim(self) -> list:
        """
        Returns the size of the bounding box in row-major order (Z, Y, X)
        :return: Size of the bounding box in row-major order (Z, Y, X)
        :rtype: list
        """
        return self.getSize().getComponents()[::-1]

    def isDisjoint(self, other) -> bool:
        """
        Determines whether two bounding boxes are disjoint from each other
        :param other: The other bounding box for comparison
        :type other: BoundingBox
        :return: True if the bounding boxes are disjoint, false otherwise
        :rtype: bool
        """
        if not isinstance(other, BoundingBox):
            raise ValueError("other must be a vector instead other is "
                             "{}".format(type(other)))

        result = any(r1 > r2 for r1, r2 in zip(self.getEdges()[0],
                                               other.getEdges()[1]))
        result |= any(r1 < r2 for r1, r2 in zip(self.getEdges()[1],
                                                other.getEdges()[0]))

        return result

    def isSubset(self, other):
        """
        Determines whether the bounding box is a subset of the other
        :param other: The other bounding box for comparison
        :type other: BoundingBox
        :return: True if the bounding box is a subset of the other, false
 otherwise
        :rtype: bool
        """
        if not isinstance(other, BoundingBox):
            raise ValueError("other must be a vector instead other is "
                             "{}".format(type(other)))

        # Determines whether the first bounding box's components are
        # less than or equal to the other's components
        result = any(r1 <= r2 for r1, r2 in zip(self.getEdges()[1],
                                                other.getEdges()[1]))

        # Determines whether the first bounding box's components are
        # greater than or equal to the other's components
        result &= any(r1 >= r2 for r1, r2 in zip(self.getEdges()[0],
                                                 other.getEdges()[0]))

        return result

    def isSuperset(self, other):
        """
        Determines whether the bounding box is a super set of the other
        :param other: The other bounding box for comparison
        :type other: BoundingBox
        :return: True if the bounding box is a super set of the other, false
 otherwise
        :rtype: bool
        """
        return other.isSubset(self)

    def intersect(self, other):
        """
        Returns the bounding box given by the intersection of two
bounding boxes
        :param other: The other bounding box for the intersection
        :type other: BoundingBox
        :return: An bounding box intersecting the two bounding boxes
        :rtype: BoundingBox
        """
        if self.isDisjoint(other):
            raise ValueError("The bounding boxes must not be disjoint")

        # The first edge contains the largest components of the first
        # edge of the two bounding boxes
        edge1 = Vector(*map(lambda x, y: max(x, y),
                            self.getEdges()[0].getComponents(),
                            other.getEdges()[0].getComponents()))

        # The second edge contains the smallest components of the second
        # edge of the two bounding boxes
        edge2 = Vector(*map(lambda x, y: min(x, y),
                            self.getEdges()[1].getComponents(),
                            other.getEdges()[1].getComponents()))

        return BoundingBox(edge1, edge2)

    def __str__(self):
        edge1, edge2 = self.getEdges()
        return "Edge1: {}\nEdge2: {}".format(edge1, edge2)

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise ValueError("other must be a vector instead other is "
                             + "{}".format(type(other)))

        edge1, edge2 = self.getEdges()
        result = BoundingBox(edge1 + other, edge2 + other)
        return result

    def __sub__(self, other):
        return self.__add__(other*-1)

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            raise ValueError("other must be a BoundingBox")

        s_edge1, s_edge2 = self.getEdges()
        o_edge1, o_edge2 = other.getEdges()

        return (s_edge1 == o_edge1) and (s_edge2 == o_edge2)

    def __ne__(self, other):
        return not (self == other)

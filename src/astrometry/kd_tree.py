"""
KD Tree implementation for efficient spatial searches for our geometric hash codes.

Authors: Peter Thomas
Date: 2025-10-15
"""
import numpy as np
from typing import Tuple
from ABC import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def __init__(self):
        pass


class InternalNode(Node):
    left_child: Node = None
    right_child: Node = None


class SplittingNode(InternalNode):
    split_dimension: int = None
    split_position: float = None

    def __init__(self):
        pass


class BoundingBoxNode(InternalNode):
    lower_bounds: np.ndarray = None
    upper_bounds: np.ndarray = None

    def __init__(self):
        pass


class LeafNode(Node):
    data: np.ndarray = None
    metadata: np.ndarray = None

    def __init__(self):
        pass


class KDTree:
    root: Node = None

    def __init__(self, data: np.ndarray, metadata: np.ndarray, leaf_size: int=10, use_boxes: bool=False):
        self.root = build_node(data, metadata, leaf_size, use_boxes)


def choose_split_dimension(data) -> int:
    return np.argmax(np.maximum(data) - np.minimum(data))


def partition(
    data: np.ndarray, 
    metadata:np.ndarray, 
    dimension: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float]:

    # Find the median value in the splitting dimension
    median_value = np.median(data[:, dimension])

    # Partition the data into two subsets
    left_mask = data[:, dimension] <= median_value
    right_mask = data[:, dimension] > median_value

    left_subset = data[left_mask]
    right_subset = data[right_mask]

    left_metadata = metadata[left_mask]
    right_metadata = metadata[right_mask]

    return (left_subset, left_metadata), (right_subset, right_metadata), median_value


def build_node(data, metadata, leaf_size=10, use_boxes: bool=False) -> Node:
    # If the set of points is small enough, create a leaf node
    if len(data) <= leaf_size:
        return LeafNode(data, metadata)

    # Split points along the splitting dimension
    split_dimension = choose_split_dimension(data)
    (left_data, left_metadata), (right_data, right_metadata), split_position = partition(data, metadata, split_dimension)

    if use_boxes:
        # Create bounding box node
        n = BoundingBoxNode()
        n.lower_bounds = np.min(data, axis=0)
        n.upper_bounds = np.max(data, axis=0)

    else:
        n = SplittingNode()
        n.split_dimension = split_dimension
        n.split_position = split_position

    n.left_child = build_node(left_data, left_metadata, leaf_size, use_boxes)
    n.right_child = build_node(right_data, right_metadata, leaf_size, use_boxes)
    return n
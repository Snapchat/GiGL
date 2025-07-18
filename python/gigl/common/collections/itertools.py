from typing import TypeVar

T = TypeVar("T")


def batch(list_of_items: list[T], chunk_size: int) -> list[list[T]]:
    """Takes a list of items and batches them into provided chunk sizes.
    i.e. batch([1, 2, 3, 4, 5], 2) --> [[1, 2], [3, 4], [5]]

    Args:
        list_of_items (list[T]): The list of items to be batched
        chunk_size (int): The desired size of each batch

    Returns:
        list[list[T]]: A list of batches of items
    """
    batched_list: list[list[T]] = [
        list_of_items[i : i + chunk_size]
        for i in range(0, len(list_of_items), chunk_size)
    ]
    return batched_list

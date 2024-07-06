from typing import List


def remove_small(
        dict_elements: List[dict],
        min_length: int = 10
) -> List[dict]:
    good_elements: List[dict] = []

    for element in dict_elements:

        if len(element['text']) >= min_length:
            good_elements.append(element)

    return good_elements

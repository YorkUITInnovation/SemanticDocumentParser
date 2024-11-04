from typing import List

WINDOW_SIZE: int = 1


def window_parser(elements: List[dict]) -> List[dict]:
    dict_elements: List[dict] = []

    for idx, element_data in enumerate(elements):
        window_text = element_data['text'] = element_data['text'].strip()
        node_before: int = idx - WINDOW_SIZE
        node_after: int = idx + WINDOW_SIZE

        if node_before >= 0:
            window_text = elements[node_before]['text'].strip() + " " + window_text

        if node_after < len(elements):
            window_text = window_text + " " + elements[node_after]['text'].strip()

        element_data['metadata'] = {
            **element_data['metadata'],
            'window': window_text
        }

        dict_elements.append(element_data)

    return dict_elements

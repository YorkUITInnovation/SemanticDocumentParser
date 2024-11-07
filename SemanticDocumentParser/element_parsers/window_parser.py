from typing import List

WINDOW_SIZE: int = 1


def window_parser(elements: List[dict]) -> List[dict]:
    dict_elements: List[dict] = []
    element_max_length_for_windowing = 1000

    for idx, element_data in enumerate(elements):
        window_text = element_data['text'] = element_data['text'].strip()
        node_before: int = idx - WINDOW_SIZE
        node_after: int = idx + WINDOW_SIZE

        # If not the first node, add the text of the node before
        # Only create window if the element itself is not super large
        if node_before >= 0 and len(window_text) <= element_max_length_for_windowing:
            node_before_text: str = elements[node_before]['text'].strip()

            # Only add if the node BEFORE is not too large
            if len(node_before_text) <= element_max_length_for_windowing:
                window_text = elements[node_before]['text'].strip() + " " + window_text

        # If not the last node, add the text of the node after
        # Only create window if the element itself is not super large
        if node_after < len(elements) and len(window_text) <= element_max_length_for_windowing:

            node_after_text: str = elements[node_after]['text'].strip()

            # Only add if the node AFTER is not too large
            if len(node_after_text) <= element_max_length_for_windowing:
                window_text = window_text + " " + elements[node_after]['text'].strip()

        element_data['metadata'] = {
            **element_data['metadata'],
            'window': window_text
        }

        dict_elements.append(element_data)

    return dict_elements

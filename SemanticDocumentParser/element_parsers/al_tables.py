"""
Copied from Donald Ipperciel's Al Syllabus implementation

"""

from typing import List, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from unstructured.documents.elements import Element, Table, NarrativeText, Title

from SemanticDocumentParser.element_parsers.semantic_splitter import PARSER_GENERATED_SIGNATURE


def read_tables_bs4mp(html_text: str) -> List[pd.DataFrame]:
    """
    Parse the HTML tables with BeautifulSoup

    :param html_text: The HTML text
    :return: Pandas DFs representing tables

    CREDIT: DONALD IPPERCIEL, YORK U PROFESSOR

    """

    # Parse the HTML with BeautifulSoup
    html_text = html_text.replace("th", "td")
    soup = BeautifulSoup(html_text, 'html.parser')

    # Find all tables
    tables = soup.find_all('table')

    data_frames = []
    for table in tables:
        # Find all rows
        rows = table.find_all('tr')

        table_data = []
        for row in rows:
            cols = row.find_all('td')

            # Extract text and links
            cols_text = [col.get_text() for col in cols]
            cols_links = [col.find('a')['href'] if col.find('a') and 'href' in col.find('a').attrs else '' for col in cols]

            # Combine text and links
            cols_with_links = [f'[{text}] ({link})' if link else text for text, link in zip(cols_text, cols_links)]

            table_data.append(cols_with_links)

        # Convert table data to DataFrame
        data_frames.append(pd.DataFrame(table_data))

    return data_frames


def read_tables(html_text: str, element: Element, previous_25_elements: List[Element]) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Extract tables from HTML along with their respective titles.

    CREDIT: DONALD IPPERCIEL, YORK U PROFESSOR
    """
    # Parse the HTML content with BeautifulSoup to grab all the tables
    doc_tables_df = read_tables_bs4mp(html_text)

    # Use BeautifulSoup to find all table elements and their preceding headers
    soup = BeautifulSoup(html_text, 'html.parser')

    # Find all table elements and their preceding headings
    tables = soup.find_all('table')
    table_titles = []

    for _ in tables:

        # "previous elements" is a list of the previous 25 elements. The first element is the farthest, the last is the closest.
        # to find the heading, iterate from the CLOSEST element to the FARTHEST element.
        # Find the first heading in the previous elements. If no heading is founded, use "Untitled Table" as the title.
        heading = None

        for prev_element in reversed(previous_25_elements):
            if isinstance(prev_element, Title):
                heading = prev_element.text
                break

        # Append the heading or a placeholder if none found
        table_titles.append(heading if heading else "Untitled Table")

    # Return the extracted tables and their titles
    return doc_tables_df, table_titles


def render_tables_add_to_nodes_text(table_titles, doc_table_df) -> List[str]:
    nodes_text: List[str] = []

    for idx, title in enumerate(table_titles):

        temp_df: pd.DataFrame = (doc_table_df[idx])

        temp_text = "*" + title + "*\n "
        nb_rows = len(temp_df)
        nb_columns = len(temp_df.columns)

        for j in range(1, nb_rows):

            if temp_df.iloc[0, 0] and temp_df.iloc[j, 0]:
                temp_text += (
                        "The following " + temp_df.iloc[0, 0].strip() + ": " +
                        temp_df.iloc[j, 0].strip() + " has "
                )

            for k in range(1, nb_columns - 1):

                # Can't be null
                if not temp_df.iloc[0, k] or not temp_df.iloc[j, k] or not temp_df.iloc[0, k + 1] or not temp_df.iloc[j, k + 1]:
                    continue

                temp_text += (
                        "the following " + temp_df.iloc[0, k].strip() + ": " +
                        str(temp_df.iloc[j, k]).strip() + " has "
                )

                temp_text += (
                        "the following " + temp_df.iloc[0, k + 1].strip() + ": "
                        + str(temp_df.iloc[j, k + 1]).strip() + ", "
                )

        nodes_text.append(temp_text)

    return nodes_text


def parse_nodes_text_into_yeehaw_bonafide_elements(nodes_text: List[str], element: Element) -> List[Element]:
    nodes: List[Element] = []

    el_metadata = element.metadata
    el_metadata.signature = PARSER_GENERATED_SIGNATURE

    for node_text in nodes_text:
        nodes.append(
            NarrativeText(
                text=node_text,
                metadata=el_metadata
            )
        )

    return nodes


def al_table_parser(elements: List[Element]) -> List[Element]:
    """
    Utilizes a slightly modified version of Donald Ipperciel's table parser from the Al Syllabus project.

    [NOTE: This does NOT consume the tables. So the table elements REMAIN and must later be filtered out.]

    """

    nodes: List[Element] = []

    for idx, element in enumerate(elements):

        # Make sure we add the element back (not consume it)
        nodes.append(element)

        # No further processing unless it's a table
        if not isinstance(element, Table):
            continue

        doc_tables_df, table_titles = read_tables(
            element.metadata.text_as_html,
            element,
            previous_25_elements=elements[idx - 25:idx] if idx > 25 else elements[:idx]
        )
        rendered_tables = render_tables_add_to_nodes_text(table_titles, doc_tables_df)
        nodes.extend(parse_nodes_text_into_yeehaw_bonafide_elements(rendered_tables, element))

    return nodes

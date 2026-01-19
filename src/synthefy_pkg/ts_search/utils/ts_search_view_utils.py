from typing import Dict, List

import numpy as np
from loguru import logger

COMPILE=True

def extract_search_set(
    data: np.ndarray,
    datatype_inds_dict: Dict[str, Dict[str, int]],
    search_set: List[str],
) -> np.ndarray:
    """
    For a given combined np array of any dataset type including encoded data
    extract search set windows.
    """
    logger.info(f"{search_set=}")
    extracted_data_list = []

    for search_i in search_set:
        start_idx = datatype_inds_dict[search_i]["start"]
        end_idx = datatype_inds_dict[search_i]["end"]
        extracted_data_list.append(data[start_idx:end_idx])

        logger.info(f"start: {start_idx}; end: {end_idx}")

    extracted_np = np.concatenate(extracted_data_list, axis=0)
    return extracted_np

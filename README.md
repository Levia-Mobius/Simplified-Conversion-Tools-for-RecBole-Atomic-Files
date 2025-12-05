# Simplified Conversion Tools for RecBole Atomic Files
A simplified and improved version based on the original [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets/tree/master)/[conversion_tools](https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools)

## Overview
This repository provides a lightweight and improved version of the conversion tools originally included in the RecSysDatasets. Its primary goal is to convert public recommendation datasets, such as MovieLens and Amazon, into RecBole-compatible [Atomic Files](https://recbole.io/docs/v1.0.0/user_guide/data/atomic_files.html#additional-atomic-files)

The redesign is motivated by some issues encountered when using the original conversion_tools package, especially when processing Amazon 2023 datasets, where numerous parsing errors make direct conversion infeasible. To address these limitations, key modules were rewritten with a focus on clarity, and efficiency.

## Key Improvements
Only the following modules/`.py` files are modified (paths below refer to the original repository). These files can be directly dropped into the corresponding locations of the original conversion_tools package:

1. `src/base_dataset.py`
   - Added general preprocessing utilities for MovieLens datasets.
3. `src/extended_data.py` → `src/light_extended.py`
    - A newly implemented module: `light_extended.py`, replacing the original `extended_data.py`;
    - Currently **ONLY** supports Amazon 2023 (multiple sub-datasets with similar structure) and MovieLens (from 100k to 32M);
    - Removed nested `for` loops; all conversions are simplified;
    - Combined handling of structurally similar datasets to avoid redundant code;
    - Given the structural similarity among Amazon sub-datasets, testing has been conducted only on several subsets. If you encounter issues, please report them through GitHub: updates will follow promptly.
5. `src/utils.py`
   - Adjusted to align with the redesigned `.py` files;
   - **NOTE**: Only MovieLens and Amazon datasets are supported at the moment!!!!
7. `run.py`
   - Integrate the new lightweight modules;
   - Support richer movie metadata for MovieLens (`meta.csv`).
9. `meta.csv`
    - Supplementary metadata obtained using the TMDb API, including: Movie descriptions, Release dates and Runtime. This enhances the MovieLens item feature quality when preparing RecBole datasets.

## Notes
This project **IS NOT** an official fork; It is just an independent lightweight redesign intended to complement the original tools.

## Contact
If you find any issues or would like to request additional dataset support, please open a GitHub issue or contact me (email: ag.wrld.s@gmail.com) directly :)

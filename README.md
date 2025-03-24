# HiT&MIS L1A Converter
Convert raw data from HiT&MIS-Origin to L1A data (.nc) files. This is the first step in the HiT&MIS data analysis pipeline.

## hmsao_l1a_converter.py
accepts raw files (.fits), performs dark subtraction (.nc file with darkrate and bais info required) and line straightening (using Misdesigner, see: https://github.com/sunipkm/misdesigner.git). 

__Usage__ 


    python hmsao_l1a_converter.py [-h] [--overwrite [OVERWRITE]] [--windows [WINDOWS]] [--dark [DARK]] --model [MODEL] [--chunksize [CHUNKSIZE]] [rootdir] [dest] [dest_prefix]


__Notes__ 
- a model config file is required (.json). It can be created using MisDesigner. hmsa_origin_ship.json was created by fit_hmsao.py in https://github.com/cspatel2/hmsao-instrument.git
- It will create a new directory for each new month in the destination directory.

## process_in_batches.py
downloads raw data from mounted smb drive to local directory in batches, uses hmsa_l1A_converter.py, and then deletes raw data from local directory.


__Usage__


    python  process_in_batches.py


__Notes__
- Edit the smb drive path, local directory path, and all arguments for hmsa_l1a_converter.py command directly in the Python file before running. 

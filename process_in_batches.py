#process raw files from server 10 dirs at a time
# %%
import os
import subprocess
import shutil
import time
import numpy as np
import sys
#%%

smbdir = '/run/user/1000/gvfs/smb-share:server=10.7.3.211,share=raw/hmsorigin/'
#%%
localdir = 'rawdata'
#%%
#script inputs
converter_script = 'hmsao_l1a_converter.py'  # Path to the converter script
overwrite = False
windows = '7774,5577,6300,4278,6563,4861'
model =  'hmsa_origin_ship.json' 
destdir = '../data/l1a'  # Destination directory for converted data
chunksize = '20'  # Modify as needed (number of files per batch for the converter)
command = [
    'python', converter_script,
    '--windows',windows,
    '--model', model,
    '--chunksize', chunksize,
    localdir,
    destdir,
]


def copy_directory_with_rsync(smb_dir, local_dir):
    try:
        print(f"Copying {smb_dir} to {local_dir} using rsync...")
        subprocess.run(['rsync', '-auvz', smb_dir, local_dir], check=True)  # -a: archive, -v: verbose, -z: compression
        print(f"Successfully copied {smb_dir} to {local_dir}")
    except Exception as e:
        print(f"Error copying {smb_dir} to {local_dir}: {e}")


def delete_local_directory(dir_path):
    try:
        shutil.rmtree(dir_path)  # Delete locally processed directory
        print(f"Deleted {dir_path} locally.")
    except Exception as e:
        print(f"Error deleting {dir_path} locally: {e}")

# Main processing loop
def process_in_batches(smb_raw_dir, local_raw_dir, batch_size=10):
    # Get list of directories from the SMB raw folder
    dirlist = np.sort(os.listdir(smb_raw_dir))
    all_dirs = [os.path.join(smb_raw_dir, d) for d in dirlist if os.path.isdir(os.path.join(smb_raw_dir, d))]
    
    # Process directories in batches of batch_size
    for i in range(0, len(all_dirs), batch_size):
        batch = all_dirs[i:i + batch_size]
        
        # Copy each directory to the local raw folder using rsync
        print(f'Dowloading batch: {dirlist[i:i+batch_size]} ')
        for dir_path in batch:
            copy_directory_with_rsync(dir_path, local_raw_dir)
        
        try:
            # Start the subprocess with Popen for real-time output
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Print the output and error messages as they appear in real-time
            for line in process.stdout:
                sys.stdout.write(line)  # This will print standard output in real time

            for line in process.stderr:
                sys.stderr.write(line)  # This will print error output in real time

            # Wait for the command to complete and get the return code
            process.wait()

            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}")
            else:
                print("Command completed successfully.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        # Delete the local directory after processing
        local_dirs_to_process = [os.path.join(local_raw_dir, os.path.basename(dir_path)) for dir_path in batch]
        for local_dir_path in local_dirs_to_process:
            delete_local_directory(local_dir_path)
        
        time.sleep(2)

# Run the batch processing
process_in_batches(smbdir, localdir, batch_size=1)
# %%
# %%
# %%

# %%

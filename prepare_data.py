import os
from tqdm import tqdm

def prepare_voxceleb_data(dataset_path):
    """
    Parses the VoxCeleb directory structure to create a file list and speaker-to-integer mapping.

    Args:
        dataset_path (str): The root path to the VoxCeleb dataset (e.g., '.../dev/wav').

    Returns:
        tuple: A tuple containing:
            - list: A list of (file_path, speaker_id_str) tuples.
            - dict: A dictionary mapping string speaker IDs to integer labels.
    """
    print("\nStarting dataset preparation...")

    file_list = []

    # os.walk is a generator that explores the directory tree top-down.
    for root, dirs, files in tqdm(os.walk(dataset_path), desc="Scanning directories"):
        for file in files:
            # We only care about the audio files, which have a .wav extension.
            if file.endswith(".wav"):
                # Construct the full path to the audio file.
                full_path = os.path.join(root, file)

                # Extract the speaker_id from the path.
                # The structure is .../dataset_path/idXXXXX/video_id/utterance.wav
                # The speaker_id is the 3rd to last component of the path.
                path_parts = full_path.split(os.sep)
                speaker_id = path_parts[-3]

                file_list.append((full_path, speaker_id))

    print(f"Found {len(file_list)} audio files.")

    # Get a list of all unique speaker IDs from our file list.
    unique_speakers = sorted(list(set([speaker_id for _, speaker_id in file_list])))

    # Create the mapping from the string ID to an integer label.
    speaker_to_int = {speaker_id: i for i, speaker_id in enumerate(unique_speakers)}

    print(f"Found {len(unique_speakers)} unique speakers.")
    print("Dataset preparation complete!")

    return file_list, speaker_to_int

# This block ensures the code runs only when the script is executed directly
if __name__ == "__main__":
    # --- How to use the function on your local machine ---

    # IMPORTANT: Update this path to where your VoxCeleb 'wav' directory is located.
    #
    # Example for Windows:
    # VOXCELEB_DEV_PATH = r"C:\Users\YourUser\Documents\VoxCeleb\wav"
    #
    # Example for macOS or Linux:
    # VOXCELEB_DEV_PATH = "/home/youruser/data/VoxCeleb/wav"
    
    VOXCELEB_DEV_PATH = "D:/DL/Project/vox1_dev_wav/wav" # <-- CHANGE THIS LINE

    if not os.path.exists(VOXCELEB_DEV_PATH) or "/path/to/your" in VOXCELEB_DEV_PATH:
        print("PLEASE UPDATE THE 'VOXCELEB_DEV_PATH' VARIABLE TO YOUR ACTUAL DATASET PATH!")
    else:
        # Run the preparation function
        all_files, speaker_map = prepare_voxceleb_data(VOXCELEB_DEV_PATH)

        # --- Let's inspect the output ---
        print("\n--- Sample of the File List ---")
        # Print the first 5 entries if available
        for i in range(min(5, len(all_files))):
            print(all_files[i])

        print("\n--- Sample of the Speaker-to-Integer Map ---")
        # To avoid printing the whole map, we'll just show a few items
        for i, (speaker, label) in enumerate(speaker_map.items()):
            if i >= 5:
                break
            print(f"'{speaker}': {label}")
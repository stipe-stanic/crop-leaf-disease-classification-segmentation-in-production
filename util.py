import os

def rename_subdir(root_dir: str) -> None:
    """Renames subdirectories to standard naming convention: plant_*_disease_*"""

    for dir_name in os.listdir(root_dir):
        # Remove repetition of any word
        dir_name_target = ' '.join(sorted(set(dir_name.lower().split()), key=dir_name.lower().split().index))

        # Remove list of words
        words_to_remove = ["leaf"]
        for word in words_to_remove:
            dir_name_target = dir_name_target.replace(f'_{word}_', '_')

        # Replace spaces with underscores
        dir_name_target = dir_name_target.replace(' ', '_')

        # Rename the directory
        os.rename(os.path.join(root_dir, dir_name), os.path.join(root_dir, dir_name_target))


def print_subdir_name(root_dir: str) -> None:
    """Prints a list of subdirectories names"""

    subdirs = [os.path.basename(x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
    print(subdirs)
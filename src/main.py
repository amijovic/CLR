import sys

from test import process_directory

def main(input_dir_path, output_dir_path, algorithm, option):
    process_directory(input_dir_path, output_dir_path, algorithm, option)

if __name__ == '__main__':
    input_dir_path = sys.argv[1]
    output_dir_path = sys.argv[2]
    algorithm = sys.argv[3]
    if len(sys.argv) == 5:
        option = int(sys.argv[4])
    else:
        option = None
    
    main(input_dir_path, output_dir_path, algorithm, option)
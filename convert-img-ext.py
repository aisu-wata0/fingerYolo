#! python3
import glob, os
import errno
from os.path import basename
from PIL import Image

import argparse

def ensure_dir(file_path):
    if file_path is None:
        raise ValueError("Invalid file path")
    try:
        os.makedirs(file_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Converts images to other extensions')
    parser.add_argument('directory_in',
                        help='directory with the input image files')
    parser.add_argument('directory_out', nargs='?',
                        help='directory of the output image files, default is input directory + newextension')
    parser.add_argument('-f', required=True,
                        help='Filetype in, e.g. jpg')
    parser.add_argument('-g', default="png",
                        help='Filetype out, e.g. png')
    parser.add_argument('-v', action='store_true',
                        help='Verbose, say file input and output paths')

    # Parse arguments
    args = parser.parse_args()
    
    directoryImgs_in = args.directory_in
    directoryImgs_out = args.directory_out
    filetypeExt_in = args.f
    filetypeExt_out = args.g
    verbose = args.v
    # default directory output
    if directoryImgs_out is None:
        directoryImgs_out = directoryImgs_in.rstrip('/').rstrip('\\') + "-" + filetypeExt_out

    ensure_dir(directoryImgs_out)
    for pathFilename in glob.iglob(os.path.join(directoryImgs_in, "*." + filetypeExt_in)):
        title, ext = os.path.splitext(os.path.basename(pathFilename))
        # construct file output pathFilename
        pathFilename_out = os.path.join(directoryImgs_out, title + '.' + filetypeExt_out)

        try:
            if verbose:
                print("Generating %s for %s" % (filetypeExt_out, pathFilename))
            im = Image.open(pathFilename)
            im.save(pathFilename_out)
        except Exception as e:
            print(e)

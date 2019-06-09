import glob, os

import argparse


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Writes train.txt and test.txt for darknet')
    # Required positional argument
    parser.add_argument('directory_in',
                        help='Directory where the data will reside, relative to \'darknet.exe\'')
    parser.add_argument('directory_out',
                        help='Directory to output train.txt & test.txt')
    parser.add_argument('-f', default="png",
                        help='Filetype out, e.g. png')

    args = parser.parse_args()
    
    directory_imgs = args.directory_in
    directory_out = args.directory_out
    filetypeExt = args.f

    # Percentage of images to be used for the test set
    percentage_test = float(100/8)

    # Create and/or truncate train.txt and test.txt
    with open(os.path.join(directory_out, 'train.txt'), 'w') as file_train, open(os.path.join(directory_out, 'test.txt'), 'w') as file_test:
        # Populate train.txt and test.txt
        counter = 1
        index_test = round(100 / percentage_test)
        for pathAndFilename in glob.iglob(os.path.join(directory_imgs, "*." + filetypeExt)):
            title, ext = os.path.splitext(os.path.basename(pathAndFilename))

            writePathname = directory_imgs + "/" + title + '.' + filetypeExt + "\n"

            if counter == index_test:
                file_test.write(writePathname)
                counter = 1
            else:
                file_train.write(writePathname)
                counter = counter + 1

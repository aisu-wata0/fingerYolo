
import argparse
import os
import glob
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2


def LabelBoundingBox2Yolo(imgShape, box):
    imgWidthDiv = 1.0/imgShape[0]
    imgHeightDiv = 1.0/imgShape[1]
    centerX = (box[0] + box[1])/2.0
    centerY = (box[2] + box[3])/2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    centerX = centerX*imgWidthDiv
    width = width*imgWidthDiv
    centerY = centerY*imgHeightDiv
    height = height*imgHeightDiv
    return (centerX, centerY, width, height)


def ensure_dir(file_path):
    if file_path is None:
        raise ValueError("Invalid file path")
    try:
        os.makedirs(file_path)
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise


def yolo2Coord(imgShape, box):
    img_h, img_w = imgShape[0:2]
    x, y, width, height = box
    x1, y1 = int((x + width/2)*img_w), int((y + height/2)*img_h)
    x2, y2 = int((x - width/2)*img_w), int((y - height/2)*img_h)
    return x1, y1, x2, y2


def draw_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = yolo2Coord(img.shape, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Converts labels to yolo format')
    parser.add_argument('directory_in',
                        help='directory with the input image files')
    parser.add_argument('directory_out', nargs='?',
                        help='directory of the output image files, default is input directory + \'yolo\'')
    parser.add_argument('-f', default='mnt',
                        help='Filetype in, e.g. mnt')
    parser.add_argument('-v', action='store_true',
                        help='Verbose, say file input and output paths')
    parser.add_argument('-b', default='30',
                        help='Box size of the minutiae in pixels')
                        
    args = parser.parse_args()
    
    directoryImgs_in = args.directory_in
    directoryImgs_out = args.directory_out
    filetypeExt_in = args.f
    verbose = args.v


    classes = ["minutia"]

    try:
        boxSize = int(args.b)
    except ValueError:
        exit("-b must integer. Exit.")
    
    print('boxSize:', boxSize)

    # default directory output
    if directoryImgs_out is None:
        directoryImgs_out = directoryImgs_in.rstrip('/').rstrip('\\') + "-yolo"
    
    ensure_dir(directoryImgs_out)

    max_minutia_n = 0

    # Process
    for pathFilename in glob.iglob(os.path.join(directoryImgs_in, "*." + filetypeExt_in)):
        if verbose:
            print("Input: " + pathFilename)

        title, ext = os.path.splitext(os.path.basename(pathFilename))

        # output pathname
        pathFilename_out = os.path.join(directoryImgs_out, title + ".txt")
        if verbose:
            print("Output:" + pathFilename_out)
        
        with open(pathFilename, "r") as file, open(pathFilename_out, "w+") as file_out:
            if verbose:
                print("\t> open file: ", pathFilename)

            img_path = '%s.png' % (os.path.splitext(pathFilename)[0])
            width = 0
            height = 0
            boxes = []
            # parse lines
            ct = 0
            for line in file:
                if(len(line) < 2):
                    continue
                elems = line.split(' ')
                if len(elems) < 3:
                    continue
                # first line is just image properties
                if ct == 0:
                    ## Find out width and height of image
                    # # Use .mnt format
                    # # wrong
                    # minutia_n = int(elems[0])
                    # width = float(elems[1])
                    # height = float(elems[2])
                    ## Open image the label is about to find its width and height
                    im = Image.open(img_path)
                    width = int(im.size[0])
                    height = int(im.size[1])
                    ##
                    if verbose:
                        print('width =', width, ';','height =', height)
                    ct = ct + 1
                    continue
                ## discover class id
                # cls = "minutia"
                # if cls not in classes:
                #     exit(0)
                # class_id = classes.index(cls)
                ## only one class
                class_id = 0
                ## if mnt has class information
                if len(elems) > 3:
                    class_id = int(elems[3])

                ## 
                # minutia center position
                x = float(elems[0])
                y = float(elems[1])
                # How many pixels is the box size
                # boxSize = 40
                boxSizeHalf = float(boxSize)/2
                box = (float(x - boxSizeHalf), float(x + boxSizeHalf), float(y - boxSizeHalf), float(y + boxSizeHalf))
                bb = LabelBoundingBox2Yolo((width, height), box)
                if verbose:
                    print(bb)
                boxes.append(bb)
                file_out.write(str(class_id) + " " + " ".join([str(a) for a in bb]) + '\n')

                ct = ct + 1
            # draw_boxes(np.array(Image.open(img_path)), boxes)
            max_minutia_n = max(ct-1, max_minutia_n)

    print('maximum number of minutia:', max_minutia_n)

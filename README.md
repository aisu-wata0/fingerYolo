# Project Title

YOLO configuration for minutiae detection and evaluation.

## Getting Started

### To Train

First compile Alexey's darknet <https://github.com/AlexeyAB/darknet#requirements> then setup the image files such that they are here fingerYolo/imgs+labels-mnt/\*.png and the mnt labels in the same directory, as such: fingerYolo/imgs+labels-mnt/\*.mnt

Then run

```bash
python convert-labels-mnt-yolo.py fingerYolo/imgs+labels-mnt fingerYolo/labels-yolo-30 -b 30
```

Now copy or move the labels in the same directory as the images to train as YOLO requires it (you can make a new directory to put just the imgs and yolo labels or use the same "imgs+labels-mnt")

Then run

```bash
python fingeryolo/train.py   ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30/   -tr 0.025 1.0 0.025 -e  2>&1 | tee train-$(date +%Y%m%d-%H%M).log
```

To start training the configuration in `./fingeryolo/testsOpen/v3-spp2-anchor1-box-30/v3-spp2-anchor1-box-30.cfg` using the files listed in`./fingeryolo/testsOpen/train.txt` and `./fingeryolo/testsOpen/train.txt` as the training and testing files respectively.
The weights are saved in `fingeryolo/testsOpen/v3-spp2-anchor1-box-30/weights/`
This will also evaluate the network when the training ends, if you don't want that don't use the -t or -tr options.

## Running the tests

```bash
python fingeryolo/train.py  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40   -tr 0.025 1.0 0.025 -e
```

Will run the evaluation for the network configuration listed, it automatically saved the network predictions so you don't have to run again if you want to compare the results, if you want to re-predict use the no evaluation cache option `-nec`.

This compares all network configurations in testsOpen and makes precision-recall graph comparing them

```bash
python fingeryolo/train.py   ./fingeryolo/testsOpen/*  -tr 0.025 1.0 0.025 -e
```

#! bash
python fingeryolo/train.py ./fingeryolo/testsClosed/v2-yolo-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30/  ./fingeryolo/testsClosed/v3-tiny-anchor1-box-30/  ./fingeryolo/testsClosed/v2-tiny-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40/  ./fingeryolo/testsClosed/v3-yolo-anchor1-box-30/  ./fingeryolo/testsClosed/v2-voc-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-16/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-60/ -tr 0.05 1.0 0.05 --dry | tee train-testsClosed-$(date +%Y%m%d-%H%M).log

python fingeryolo/train.py ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30/ --dry | tee train-20190608-v3-spp2-anchor1-box-30.log

python fingeryolo/train.py ./fingeryolo/testsClosed/v2-yolo-anchor5-box-30/  ./fingeryolo/testsClosed/v3-tiny-anchor1-box-30/  ./fingeryolo/testsClosed/v2-tiny-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40/  ./fingeryolo/testsClosed/v3-yolo-anchor1-box-30/  ./fingeryolo/testsClosed/v2-voc-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-16/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-60/ -tr 0.05 1.0 0.05 --dry | tee train-testsClosed-$(date +%Y%m%d-%H%M).log


./darknet detector test ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30//../obj.data ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30//v3-spp2-anchor0-box-30.cfg ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30//weights\v3-spp2-anchor0-box-30-train_last.weights -ext_output  -thresh 0.5


./darknet detector test ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30//../obj.data ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30//v3-spp2-anchor1-box-30.cfg ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30//weights\v3-spp2-anchor1-box-30-train_last.weights -ext_output  -thresh 0.5

python fingeryolo/train.py  ./fingeryolo/testsClosed/v2-yolo-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30/  ./fingeryolo/testsClosed/v3-tiny-anchor1-box-30/  ./fingeryolo/testsClosed/v2-tiny-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40/  ./fingeryolo/testsClosed/v3-yolo-anchor1-box-30/  ./fingeryolo/testsClosed/v2-voc-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-16/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-60/  -tr 0.05 1.0 0.05  | tee train-20190608-2005.log

python fingeryolo/train.py ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30/  -tr 0.05 1.0 0.05 -e  | tee train-$(date +%Y%m%d-%H%M).log




python fingeryolo/train.py ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30/  -tr 0.05 1.0 0.05 -e  | tee train-$(date +%Y%m%d-%H%M).log






python fingeryolo/train.py ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30/  -tr 0.025 1.0 0.025 -e 2>&1 | tee train-$(date +%Y%m%d-%H%M).log

python fingeryolo/train.py  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-16  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-60        ./fingeryolo/testsOpen/v2-voc-anchor5-box-30  ./fingeryolo/testsOpen/v3-spp2-anchor0-box-30  ./fingeryolo/testsOpen/v3-spp2-anchor1-box-16  ./fingeryolo/testsOpen/v3-spp2-anchor1-box-40  ./fingeryolo/testsOpen/v3-spp2-anchor1-box-60  -tr 0.025 1.0 0.025  2>&1 | tee train-$(date +%Y%m%d-%H%M).log


python fingeryolo/train.py   ./fingeryolo/testsOpen/v3-yolo-anchor1-box-30   ./fingeryolo/testsOpen/v3-spp2-anchor1-box-40    ./fingeryolo/testsOpen/v3-spp2-anchor1-box-60   ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40/    ./fingeryolo/testsClosed/v3-spp2-anchor1-box-60/  -tr 0.025 1.0 0.025  2>&1 | tee train-$(date +%Y%m%d-%H%M).log


python fingeryolo/train.py    ./fingeryolo/testsClosed/v3-spp2-anchor1-box-60/  -tr 0.025 1.0 0.025  -e  2>&1 | tee train-$(date +%Y%m%d-%H%M).log

python fingeryolo/train.py   ./fingeryolo/testsOpen/* ./fingeryolo/testsClosed/*   -tr 0.025 1.0 0.025 -e  2>&1 | tee train-$(date +%Y%m%d-%H%M).log



python fingeryolo/train.py   ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30/   -tr 0.025 1.0 0.025 -e 


cd /e/Users/Serbena/Git/TG/darknet/build/darknet/x64/fingeryolo/testsClosed/v3-spp2-anchor1-box-40/preds
mv pred-t0.050 pred-t0.050-bugged

python fingeryolo/train.py   ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40   -tr 0.025 1.0 0.025 -e  2>&1 | tee train-$(date +%Y%m%d-%H%M).log


py fingeryolo/template.py ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30/preds/pred-t0.400/

python fingeryolo/train.py   ./fingeryolo/testsOpen/*  -tr 0.025 1.0 0.025 -e
python fingeryolo/train.py   ./fingeryolo/testsClosed/*  -tr 0.025 1.0 0.025 -e

# compare closed with open dataset networks
python fingeryolo/train.py   ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30 ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30 ./fingeryolo/testsClosed/v3-spp2-anchor1-box-16/ ./fingeryolo/testsOpen/v3-spp2-anchor1-box-16/  -tr 0.025 1.0 0.025 -e 


# prepare database

for dir in "FVC2004DB2" "FVC2004DB3" "FVC2004DB4" ; do python convert-img-ext.py -f bmp "$dir" "$dir" ; done

for dir in "FVC2004DB2" "FVC2004DB3" "FVC2004DB4" ; do python convert-labels-mnt-yolo.py "$dir" "$dir" ; done


python fingeryolo/train.py   ./fingeryolo/testsOpenFull/*  2>&1 | tee train-$(date +%Y%m%d-%H%M).log

py fingeryolo/template.py ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30/preds/pred-t0.200/

for dir in "labels/FM3_FVC2004DB1A_MNT" "labels/FM3_FVC2004DB3A_MNT" ; do python convert-labels-mnt-yolo.py "$dir" "$dir" ; done


for dir in "FVC2004DB4" "FVC2004DB2"  ; do mkdir "${dir}_old"; mv ${dir}/*.bmp "${dir}_old/"; mv ${dir}/*.mnt "${dir}_old/" ; done


python fingeryolo/train.py ./fingeryolo/testsOpenFM3/*  2>&1 | tee train-$(date +%Y%m%d-%H%M).log


cd finger-matcher
./run.sh ../testsOpen/v3-spp2-anchor1-box-30/preds/pred-t0.200-afis  2>&1 | tee ROC-$(date +%Y%m%d-%H%M).log
cd ..


py finger-matcher.py ../testsOpen/v3-spp2-anchor1-box-30/preds/pred-t0.425   2>&1 | tee finger-matcher/logs/ROC-$(date +%Y%m%d-%H%M).log
py plt_match_curve.py -g finger-matcher/genuine_scores.txt -i finger-matcher/impostor_scores.txt

py plt_match_curve_batch.py ./testsOpen/v3-spp2-anchor1-box-30/preds/pred-t*[^a-z]   2>&1 | tee logs/ROC-$(date +%Y%m%d-%H%M).log


python fingeryolo/train.py ./fingeryolo/testsOpenFM3/* -tr 0.050 1.0 0.050 -e --dry 2>&1 | tee fingeryolo/logs/train-$(date +%Y%m%d-%H%M).log
 

#   ./fingeryolo/testsOpenFM3/* -> sourceAfis -> ROCcuve

py plt_match_curve_batch.py ./testsOpenFM3/v2-yolo-anchor5-box-30/preds/pred-t*[^a-z] --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log
py plt_match_curve_batch.py ./testsOpenFM3/v3-spp2-anchor2-box-30/preds/pred-t*[^a-z] --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log
py plt_match_curve_batch.py ./testsOpenFM3/v3-spp3-anchor2-box-30/preds/pred-t*[^a-z] --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log
py plt_match_curve_batch.py ./testsOpenFM3/v3-tiny-anchor1-box-30/preds/pred-t*[^a-z] --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log
py plt_match_curve_batch.py ./testsOpenFM3/v3-yolo-anchor1-box-30/preds/pred-t*[^a-z] --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log

# ./testsOpenFM3/*/preds/pred-t*[^a-z]



py plt_match_curve_batch.py  ./testsOpenFM3/v2-yolo-anchor5-box-30/preds/pred-t*[^a-z]   ./testsOpenFM3/v3-spp2-anchor2-box-30/preds/pred-t*[^a-z]  ./testsOpenFM3/v3-spp3-anchor2-box-30/preds/pred-t*[^a-z]  ./testsOpenFM3/v3-tiny-anchor1-box-30/preds/pred-t*[^a-z]  ./testsOpenFM3/v3-yolo-anchor1-box-30/preds/pred-t*[^a-z] --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log


py plt_match_curve_batch.py  ./testsOpenFM3/v3-spp2-anchor2-box-30/preds/pred-t*[^a-z]  --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log


py plt_match_curve_batch.py  ./testsOpen/v3-spp2-anchor1-box-30/preds/pred-t*[^a-z]  --img_dir imgs 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log

# changed finger-matcher main to receive list of files instead of directory as argument
# /e/Users/Serbena/Git/TG/darknet/build/darknet/x64
cd ..
java -jar ./fingeryolo/finger-matcher/target/finger-matcher-1.0-SNAPSHOT-jar-with-dependencies.jar "./fingeryolo/testsOpenFM3/test.txt"
# mv genuine_scores impostor_scores to fingeryolo

# RELEVANT
py plt_match_curve_batch.py  ./testsOpenFM3/v2-yolo-anchor5-box-30/preds/pred-t*[^a-z]   ./testsOpenFM3/v3-spp2-anchor2-box-30/preds/pred-t*[^a-z]   ./testsOpenFM3/v3-tiny-anchor1-box-30/preds/pred-t*[^a-z]  --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log

# changed back: finger-matcher main to receive directory instead of list of files as argument
py plt_match_curve_batch.py  ./testsOpen/*/preds/pred-t*[^a-z]     --img_dir imgs    2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log


sed -i -r 's/"direction": [0-9]\.[^ ]*[0-9], "typ/"direction": 0.0, "typ/g' *

py plt_match_curve_batch.py  ./testsOpenFM3/v2-yolo-anchor5-box-30/preds/pred-t*[^a-z]   ./testsOpenFM3/v3-spp2-anchor2-box-30/preds/pred-t*[^a-z]   ./testsOpenFM3/v3-tiny-anchor1-box-30/preds/pred-t*[^a-z]   ./testsOpen/v2-yolo-anchor5-box-30/preds/pred-t*[^a-z]     ./testsOpen/v2-voc-anchor5-box-30/preds/pred-t*[^a-z]   ./testsOpen/v3-spp2-anchor1-box-30/preds/pred-t*[^a-z]   ./testsOpen/v3-yolo-anchor1-box-30/preds/pred-t*[^a-z]    --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log


py plt_match_curve_batch.py   ./testsOpen/v2-voc-anchor5-box-30/preds/pred-t0.100 ./testsOpen/v2-yolo-anchor5-box-30/preds/pred-t0.100 ./testsOpen/v3-spp2-anchor0-box-30/preds/pred-t0.100 ./testsOpen/v3-spp2-anchor1-box-16/preds/pred-t0.100 ./testsOpen/v3-spp2-anchor1-box-30/preds/pred-t0.100 ./testsOpen/v3-spp2-anchor1-box-40/preds/pred-t0.100 ./testsOpen/v3-spp2-anchor1-box-60/preds/pred-t0.100 ./testsOpen/v3-tiny-anchor1-box-30/preds/pred-t0.100 ./testsOpen/v3-yolo-anchor1-box-30/preds/pred-t0.100  --img_dir imgs    2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log
 


py plt_match_curve_batch.py  ./testsOpenFM3/v3-spp2-anchor2-box-30/preds/pred-t0.10*[^a-z]  --img_dir imgs_FM3 2>&1  | tee logs/ROC-$(date +%Y%m%d-%H%M).log
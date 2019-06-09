python fingeryolo/train.py ./fingeryolo/testsClosed/v2-yolo-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30/  ./fingeryolo/testsClosed/v3-tiny-anchor1-box-30/  ./fingeryolo/testsClosed/v2-tiny-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40/  ./fingeryolo/testsClosed/v3-yolo-anchor1-box-30/  ./fingeryolo/testsClosed/v2-voc-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-16/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-60/ -tr 0.05 0.951 0.05 --dry | tee train-testsClosed-$(date +%Y%m%d-%H%M).log

python fingeryolo/train.py ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30/ --dry | tee train-20190608-v3-spp2-anchor1-box-30.log

python fingeryolo/train.py ./fingeryolo/testsClosed/v2-yolo-anchor5-box-30/  ./fingeryolo/testsClosed/v3-tiny-anchor1-box-30/  ./fingeryolo/testsClosed/v2-tiny-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40/  ./fingeryolo/testsClosed/v3-yolo-anchor1-box-30/  ./fingeryolo/testsClosed/v2-voc-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-16/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-60/ -tr 0.05 0.951 0.05 --dry | tee train-testsClosed.log


./darknet detector test ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30//../obj.data ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30//v3-spp2-anchor0-box-30.cfg ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30//weights\v3-spp2-anchor0-box-30-train_last.weights -ext_output  -thresh 0.5


./darknet detector test ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30//../obj.data ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30//v3-spp2-anchor1-box-30.cfg ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30//weights\v3-spp2-anchor1-box-30-train_last.weights -ext_output  -thresh 0.5

python fingeryolo/train.py  ./fingeryolo/testsClosed/v2-yolo-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30/  ./fingeryolo/testsClosed/v3-tiny-anchor1-box-30/  ./fingeryolo/testsClosed/v2-tiny-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor0-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-40/  ./fingeryolo/testsClosed/v3-yolo-anchor1-box-30/  ./fingeryolo/testsClosed/v2-voc-anchor5-box-30/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-16/  ./fingeryolo/testsClosed/v3-spp2-anchor1-box-60/  -tr 0.05 1.0 0.05  | tee train-20190608-2005.log

python fingeryolo/train.py ./fingeryolo/testsClosed/v3-spp2-anchor1-box-30/  -tr 0.05 1.0 0.05  | tee train-$(date +%Y%m%d-%H%M).log
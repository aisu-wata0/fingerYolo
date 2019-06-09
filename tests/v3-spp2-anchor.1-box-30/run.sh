python convert-labels-mnt-yolo.py -f mnt imgs+labels-mnt labels-yolo-30 -b 30
cd ..
./darknet.exe detector train fingeryolo/obj.data "fingeryolo/yolov3-spp-finger.2.cfg"  -map
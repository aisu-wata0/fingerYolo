python convert-labels-mnt-yolo.py -f mnt imgs+mnt-yolo-40 imgs-yolo-40 -b 40
./darknet.exe detector train fingeryolo/obj.data "fingeryolo/yolov3-spp-finger.2.cfg"  -map
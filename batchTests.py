

import subprocess
import glob

if __name__ == "__main__":
	import argparse
	## Instantiate the parser
	parser = argparse.ArgumentParser(
	description='Runs multuple predictions to graph results\n'
	'python ' + __file__,
	formatter_class=argparse.RawTextHelpFormatter)
	
	parser.add_argument('-v', action='store_true',
	help='verbose')
	parser.add_argument('-d', '--dry', action='store_true',
							help="dry run")
	## Instantiate the parser
	
	## Parse arguments
	args = parser.parse_args()


	command = './darknet.exe detector train fingeryolo/obj.data "fingeryolo/yolov3-spp-finger.2.cfg"  -map'

	commandTest = 'py ./fingeryolo/predict.py ./fingeryolo/obj.data ./fingeryolo/v3-spp2-anchor.1-box-30/v3-spp2-anchor.1-box-30.cfg ./fingeryolo/v3-spp2-anchor.1-box-30/weights/yolov3-spp-finger_4000.weights ./fingeryolo/test.txt -dont_show -save_labels -tr 0.025 0.951 0.025 -c'

	testConfigs = []
	for boxSize in ['16', '30', '40', '60']:
		testConfigName = 'v3-spp2-anchor.1-box-{}'
		testConfigs.append((testConfigName, boxSize))

	# boxSize
	boxSize = 16

	for testConfigName in ['v3-spp-anchor.1-box-{}', 'v3-tiny-anchor.0-box-{}', 'v3-tiny-anchor.1-box-{}',  'v3-tiny-anchor.1-lrConst-box-{}', 'v3-yolo-anchor.1-box-{}']:
		testConfigs.append((testConfigName, boxSize))
	
	cfgs = ['v2-tiny-anchor.5-box-{}',
			  'v2-voc-anchor.5-box-{}',  'v2-yolo-anchor.5-box-{}']
	for testConfigName in cfgs:
		testConfigs.append((testConfigName, boxSize))

	cfgs = ['v3-spp-anchor.1-box-{}', 'v3-tiny-anchor.0-box-{}', 'v3-tiny-anchor.1-box-{}',
			  'v3-tiny-anchor.1-lrConst-box-{}', 'v3-yolo-anchor.1-box-{}']
	for testConfigName in cfgs:
		testConfigs.append((testConfigName, boxSize))
	# boxSize


	for testConfigName, boxSize, in testConfigs:
		testConfigName = testConfigName.format(boxSize)
		pathDir = './fingeryolo/tests'
		pathDirCfg = '{}/{}'.format(pathDir, testConfigName)
		pathWeight = sorted(list(glob.iglob(pathDirCfg + '/weights/*.weights')))[-1]

		commandTest = f'py ./fingeryolo/predict.py {pathDir}/obj.data {pathDirCfg}/{testConfigName}.cfg {pathWeight} {pathDir}/test.txt -dont_show -save_labels -tr 0.025 0.951 0.025 -c'
		print("calling ", commandTest, flush=True)
		#
		# call predict.py to use network and make curve

		if not args.dry:
			subprocess.call(commandTest)
		#

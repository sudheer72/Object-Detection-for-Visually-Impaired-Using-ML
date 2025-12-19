YOLO (You Only Look Once)
you need to import the below files to configure the model

1. 'coco.names' file contains all the classes(objects) that this model is trained to detect.
   To see the contents of this file after importing the essentials open this file 
   
   with open(coco.names) as f:
	for object in f.strip():
		print(object)

2. 'yolov3.weights' this file contains all the weights of the nodes of neural layers

3. 'yolov3.cfg' this file contains the biases of nodes or neurons


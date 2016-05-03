# neural_art_tensorflow
Firstly, we have to install TensorFlow on Ubuntu 14.04 by fllowing the instruction: https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#download-and-setup

Clone the project: https://github.com/tunglx/neural_art_tensorflow 

Download the VGG and I2V models: https://goo.gl/hKCr8e 

Running “neural_art_tensorflow.py” without options yields the default settings and input images. Available options are:

“-m, --model”: Model type: “VGG” or “I2V”

“-p, --model_path”: Path to the Model

“-cp, --content_path”: Path to the Content image

“-sp, --style_path”: Path to the Style image

“-i, --iterations”: Number of iterations

“-a, --alpha”: alpha (content weight) 

“-b, --beta”: beta (style weight)

For example:

python neural_art_tensorflow.py -m VGG -p ./VGG_model -cp ./images/samsung.jpg -sp ./images/style.jpg -i 100 -a 1 -b 200

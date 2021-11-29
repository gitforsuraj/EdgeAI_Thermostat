# EdgeAI_Thermostat  

Python Version: Python 3.9

Python libraries to install:
1) Tensorflow
2) Keras
3) Pandas
4) Numpy

Other software to install:
1) Arduino IDE (Note: Check to make sure the correct comm port is selected!)
2) Arduino CLI
3) Tera Term Data Logger (Note: Check to make sure the correct comm port is selected!)

Arduino libraries to install: 
1) Wire.h
2) RTClib.h
3) Adafruit_GFX.h
4) Adafruit_SSD1306.h
5) Arduino_HTS221.h
6) TensorFlowLite.h (Version 1.15)

Setup the project:
1) Create main directory "EDGEAI_THERMOSTAT"
2) Create sub directory "Models"
3) Create sub directory "Datasets"
4) Save "dataset_v1.1.csv" in "Datasets" directory
5) Save "model_v1.2.py" in "Models" directory
6) Save "thermo_final.ino" in "EDGEAI_THERMOSTAT" directory

Run the project:
1) Open terminal to "EDGEAI_THERMOSTAT" directory 
2) Enter the following: while true; do python3 model_v1.2.py; sleep 604800; done
3) Done!

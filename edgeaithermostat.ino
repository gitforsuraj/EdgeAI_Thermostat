/*
  Referenced:
   https://github.com/arduino/ArduinoTensorFlowLiteTutorials/blob/master/GestureToEmoji/ArduinoSketches/IMU_Classifier/IMU_Classifier.ino
   https://www.digikey.com/en/maker/projects/intro-to-tinyml-part-2-deploying-a-tensorflow-lite-model-to-arduino/59bf2d67256f4b40900a3fa670c14330  
   https://forum.arduino.cc/t/rtc-clock-with-days-of-week/426045/2
 */

/*
 - Timer 1 is used for model suspension for 1 hr
 - Timer 2 is used for weekly print to csv and and reflash
 - current version runs with no issues. 
 - datapoint object works. 
 - used array for storing datapoint objects.
 - added operation mode selection: off, cool, heat, auto.
    - cool will trigger when passed 2 degrees and overshoot target by 1 degree.
    - heat will trigger when passed 2 degrees and overshoot target by 1 degree.
    - auto will trigger when above or below by 2 degrees, and overshoot by 1 degree.
 - some functions/vars reorganized due to running out of memory
 - all Serial prints have been commented out, since serial print is also how data is sent to csv
 */

/*
  Update: 5/4/2021
  - Target temp as uint8_t, instead of float
  - User input now suspends model usage for 1 hr
  - Timer 2 used for model suspension
  - Timer set accounts for edge case where current hour is 23, so next hour is 0
  - Will print "Timer set." in yellow section of display when timer is set
      - just for debug, will remove later. or change message. 
  - Found code which crashes MBed OS, made comments and avoided crashes
  - Added more useful comments
  - Errors that are hopefully fixed: (but not confirmed)
      - time on display not updating while timer set
      - "Timer set." text not cleared from display after timer fires
      - MBed OS crashes
      - exiting user control mode and re-entering model mode after timer expires
      - clearing timer
 */

#include <Wire.h>
#include "RTClib.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Arduino_HTS221.h>

// TFL stuff
// Standard to any TFL project
// using version 1.15.0 of TensorFlow library
#include "TensorFlowLite.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"     // contains some operations we use
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"  // debugging
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"     // runs inference engine
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"// register specific TFL operations
#include "tensorflow/lite/version.h"

// Model
#include "thermo_model.h"

float const ADJ = 4.00;

RTC_DS3231 rtc;       // rtc module
byte target;          // target temperature thermostat is 'set' to. DO NOT assign a value here, for some reason it causes MBedOS to crash.
float temp, humidity;
byte heatmode = 0;    // 0 = off, 1 = ac, 2 = heat, 3 = auto
byte autocool = 0;
byte autoheat = 0;
byte pr2, pr3, pr4 = 0;    // bytes (either 1 or 0) to avoid polling button more than once per press
byte model_mode;      // byte (either 1 or 0) to record if using user input or using model for target temp. **DO NOT ASSIGN HERE**
byte timer_set;       // byte to keep track of if timer is set or not. **DO NOT ASSIGN HERE**
int day = 0;          // Day of Week, values 0-6 (0 == Sun, 6 == Sat)
DateTime now;         // Object for holding Time object retrieved from RTC module
DateTime useradj;     // Object for holding Time object when user made adjustment to target temperature

#define OLED_RESET 4  // not used, but required for declaration
Adafruit_SSD1306 display(128, 64, &Wire, OLED_RESET); // display res 128x64, i2c

class datapoint {
  private:
    byte dptarget;
    int dphumidity;
    int dpmo;
    int dpdow;
    int dphr;
  public:
    datapoint(){
      int adjustment, mm, yy;
      int year = now.year();
      this->dpmo = now.month();
      int day = now.day();
      if (year<2000) year+=2000;
      adjustment = (14 - dpmo) / 12;
      mm = dpmo + 12 * adjustment - 2;
      yy = year - adjustment;
      this->dpdow = (day + (13 * mm - 1) / 5 + yy + yy / 4 - yy / 100 + yy / 400) % 7;
    
      this->dptarget = target;
      this->dphumidity = (int)humidity;
      this->dphr = now.hour() - 1; // minus one since we are recording the data point 1 hr after it was requested.
    }
    void printdp(){
      Serial.print(dpmo);
      Serial.print(", ");
      Serial.print(dpdow);
      Serial.print(", ");
      Serial.print(dphr);
      Serial.print(", ");
      Serial.print(dphumidity);
      Serial.print(", ");
      Serial.print(dptarget);
      Serial.print("\n");
    }
};

datapoint* dparr[168] = {};
byte arrindex = 0;

void printarr()
{
  Serial.println("Month,Day,Time of Day,Relative Humidity,Target Temperature");
  for (int i = 0; i < arrindex; i++)
  {
    dparr[i]->printdp();
  }
}

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  /*
    Arena: create an area of memory to use for input, output, and other TensorFlow 
    arrays. You'll need to adjust this by compiling, running, and looking for errors.
  */
  constexpr int kTensorArenaSize = 8 * 1024;      // 1 KB to start with - can probably be between 3 and 5 KB instead.
  uint8_t tensor_arena[kTensorArenaSize];
} // TFL Namespace 

void loop() {
  now = rtc.now();
  day = weekday((int)now.year(), (int)now.month(), (int)now.day());
  temp = HTS.readTemperature(FAHRENHEIT) - ADJ;
  humidity = HTS.readHumidity();
  
  /*************************/
  /* Starting TFL magic... */
  /*************************/
    // Copy value to input buffer (tensor)
    model_input->data.f[0] = now.month();
    model_input->data.f[1] = day;
    model_input->data.f[2] = now.hour();
    model_input->data.f[3] = (int)humidity;
  
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on input\n");
    }

    // Read predicted y value from output buffer (tensor)
    target = (byte)model_output->data.f[0];
  /********************/
  /* End of TFL magic */
  /********************/
  
  printinfo(target); // print to display function
  display.display();  // flushes changes to display
  userin(); // get user input
  if(rtc.alarmFired(2)) // check if weekly alarm has fired. if so, print linked list
    {
      rtc.clearAlarm(2);
      // call print data points here since weekly alarm has expired.
      printarr();
    }

  while(model_mode == 0)    // Loop for when model is not being used due to user input
  {
    now = rtc.now();
    day = weekday((int)now.year(), (int)now.month(), (int)now.day());
    temp = HTS.readTemperature(FAHRENHEIT) - ADJ;
    humidity = HTS.readHumidity();

    printinfo(target);
    display.display();  // flushes changes to display
    userin();
    
    if(rtc.alarmFired(1)) // Check if alarm has expired, if so re-enable model_mode
    {
      datapoint *d = new datapoint();
      dparr[arrindex] = d;
      arrindex++;
      
      rtc.clearAlarm(1); // clear 1 hr timer.
      model_mode = 1; // ready to go back into model mode.
      timer_set = 0;
      display.clearDisplay();
      printinfo(target);
      display.display();
    }
    if(rtc.alarmFired(2)) // weekly timer could fire while not in model mode. check weekly timer. 
    {
      rtc.clearAlarm(2);
      // call print data points here
      printarr();
    }
  }
}


/*
  ********************************
  Further functions defined below: 
  ********************************
 */


void proginit()
{
  Wire.begin();   // I2C
  Serial.begin(9600);

  pinMode(2, INPUT); // temp down
  pinMode(3, INPUT); // temp up
  pinMode(7, INPUT); // mode select
  pinMode(6, OUTPUT); // AC LED
  pinMode(5, OUTPUT); // Heat LED

  digitalWrite(6, LOW); // AC OFF
  digitalWrite(5, LOW); // HEAT OFF
  
  if (! rtc.begin())
  {
    //Serial.println("Couldn't find RTC");
    while(1);
  }

  rtc.disable32K();
  rtc.clearAlarm(1);

  if(!rtc.setAlarm2(
            rtc.now() + TimeSpan(120),
            DS3231_A2_Day
    )) {
        //Serial.println("Error, alarm wasn't set!");
    }else {
        //Serial.println("Alarm will happen in 1 wk!");  
    }

  /*
    // Uncomment this section for first time setup. +11 is to account for compile and flash time.
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    DateTime tensec = DateTime(rtc.now().unixtime()+11);
    rtc.adjust(tensec);
  */

  if (rtc.lostPower())
  {
    //Serial.println("RTC lost power, set time.");

    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }
  
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);  // Display
  
  if (!HTS.begin()) // Error checking display
  {
    //Serial.println("Failed to initialize Humidity/Temp Sensor.");
    while(1);
  }
}

 
void userin()
{
  if(!digitalRead(2) && pr2 == 0)
  {
    target--;
    pr2 = 1; // p2 means 'pressed 2' (currently pressed)
    
    // stop using ML model
    model_mode = 0;

    rtc.clearAlarm(1);
      
    if(!rtc.setAlarm1(
            rtc.now() + TimeSpan(3600),
            DS3231_A1_Hour 
    )) {
        //Serial.println("Error, alarm wasn't set!");
    }else {
        //Serial.println("Alarm will happen in 1 hr!");  
    }
    
    timer_set = 1;
  }
  if(digitalRead(2) && pr2 == 1)
  {
    pr2 = 0; // p2 no longer pressed
  }

  
  if(!digitalRead(3) && pr3 == 0)
  {
    target++;
    pr3 = 1; // p3 means 'pressed 3' (currently pressed)
    
    // stop using ML model
    model_mode = 0;

    rtc.clearAlarm(1);

    if(!rtc.setAlarm1(
            rtc.now() + TimeSpan(3600),
            DS3231_A1_Hour 
    )) {
        //Serial.println("Error, alarm wasn't set!");
    }else {
        //Serial.println("Alarm will happen in 1 hr!");  
    }
    
    timer_set = 1; 
  }
  if(digitalRead(3) && pr3 == 1)
  {
    pr3 = 0; // p3 no longer pressed
  }
  if(!digitalRead(7) && pr4 == 0)
  {
    heatmode++;
    if (heatmode == 4) heatmode = 0;
    pr4 = 1;
  }
  if (digitalRead(7) && pr4 == 1)
  {
    pr4 = 0;
  }
  setLED();
}

void setLED()
{
  if (heatmode == 1 && (temp > (target + 2))) // if AC mode and hotter than target
  {
    digitalWrite(6, HIGH);
    digitalWrite(5, LOW);
  }
  else if (heatmode == 1 && (temp < (target - 1)))
  {
    digitalWrite(6, LOW);
    digitalWrite(5, LOW);
  }
  else if (heatmode == 2 && (temp < (target - 2))) // if heat mode enabeled and colder than target
  {
    digitalWrite(5, HIGH);
    digitalWrite(6, LOW);
  }
  else if (heatmode == 2 && (temp > (target + 1)))
  {
    digitalWrite(5, LOW);
    digitalWrite(6, LOW);
  }
  else if (heatmode == 3)
  {
    if ((temp > (target + 2)) && autocool == 0)
    {
      autocool = 1;
      digitalWrite(6, HIGH);
      digitalWrite(5, LOW);
    }
    else if ((temp > (target - 1)) && autocool == 1)
    {
      digitalWrite(6, HIGH);
      digitalWrite(5, LOW);
    }
    else if ((temp < (target - 2)) && autoheat == 0)
    {
      autoheat = 1;
      digitalWrite(5, HIGH);
      digitalWrite(6, LOW);
    }
    else if ((temp < (target + 1)) && autoheat == 1)
    {
      digitalWrite(5, HIGH);
      digitalWrite(6, LOW);
    }
    else
    {
      digitalWrite(5, LOW);
      digitalWrite(6, LOW);
      autocool = 0;
      autoheat = 0;
    }
  }
  else
  {
    digitalWrite(5, LOW);
    digitalWrite(6, LOW);
    autocool = 0;
    autoheat = 0;
  }
}

int weekday(int year, int month, int day)
/* Calculate day of week in proleptic Gregorian calendar. Sunday == 0. */
// from: https://forum.arduino.cc/t/rtc-clock-with-days-of-week/426045/2
{
  int adjustment, mm, yy;
  if (year<2000) year+=2000;
  adjustment = (14 - month) / 12;
  mm = month + 12 * adjustment - 2;
  yy = year - adjustment;
  return (day + (13 * mm - 1) / 5 +
    yy + yy / 4 - yy / 100 + yy / 400) % 7;
}
 
void printinfo(float tt)
{
  display.clearDisplay();
  display.setTextColor(WHITE);
  display.setTextSize(1);
  
/* Target Temp */
  display.setCursor(8,0);
  display.print("Target temp:    ");
  display.print(tt,0);
  display.print(" F");
  if (heatmode == 0) display.print("Mode: OFF");
  else if (heatmode == 1) display.print("Mode: COOL");
  else if (heatmode == 2) display.print("Mode: HEAT");
  else display.print("Mode: AUTO");
  
/* Humidity */
  display.setCursor(0,17);
  display.print("Humidity:     ");
  display.print(humidity);
  display.print(" %");

/* Measured Temp */
  display.setCursor(0,27);
  display.print("Temperature:  ");
  display.print(temp);
  display.print(" F");

/* Print DoW */
  day = weekday((int)now.year(), (int)now.month(), (int)now.day());
  display.setCursor(0,37);
  display.print("Day:        ");
  //display.print(now.getDoW());
  switch(day){
    case(0):
      display.print("   Sunday");
      break;
    case(1):
      display.print("   Monday");
      break;
    case(2):
      display.print("  Tuesday");
      break;
    case(3):
      display.print("Wednesday");
      break;
    case(4):
      display.print(" Thursday");
      break;
    case(5):
      display.print("   Friday");
      break;
    case(6):
      display.print(" Saturday");
      break;
  }

/* Time */
  display.setCursor(0,47);
  display.print("Time:           ");
  if (now.hour() < 10)
  {
    display.print(" ");
  }
  display.print(now.hour(), DEC);
  display.print(":");
  if (now.minute() < 10)
  {
    display.print("0");
  }
  display.print(now.minute(), DEC);

/* Date */
  display.print("Date:      ");
  display.print(now.year(), DEC);
  display.print("/");
  if (now.month() < 10)
  {
    display.print("0");
  }
  display.print(now.month(), DEC);
  display.print("/");
  if (now.day() < 10)
  {
    display.print("0");
  }
  display.print(now.day(), DEC);
}

void setup() 
{
  // proginit() runs setup for i2c, display, thermometer, rtc, digital pins
  proginit();
  
/*****************************/
/* TFL Initialization things */
/*****************************/
  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  // read in the model from thermo_model.h
  model = tflite::GetModel(thermo_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report("Model version does not match schema");
    while(1);
  }

  // Pull in only needed operations (should match NN layers seen in Netron graph)
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::ops::micro::Register_FULLY_CONNECTED(),
    1, 9);
  
  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, micro_mutable_op_resolver, tensor_arena, 
    kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
/*****************************/
/* TFL Initialization things */
/*****************************/

model_mode = 1;
timer_set = 0;

} // End of Setup()

// source:
// https://github.com/hamuchiwa/AutoRCCar/blob/master/arduino/rc_keyboard_control.ino

int command = 1; // go forwards by default, may change latter

// may change to fit schematic
// these are the pin numbers, that the arduino has connected to the car
int forward = 1;
int reverse = 2;
int left    = 3;
int right   = 4;

// duration for output
int time = 50;

void setup()
{
  // initialize values for forwards, back, left, right
  // pinMode(LED_BUILTIN, OUTPUT);
  pinMode(forward, OUTPUT);
  pinMode(reverse, OUTPUT);
  pinMode(left, OUTPUT);
  pinMode(right, OUTPUT);
  // initialize serial communication
  Serial.begin(9600);
}

void loop()
{
  // see if there's incoming serial data: receive command
  if (Serial.available() > 0) {
    // read the oldest byte in the serial buffer:
    // gets command from the laptop, or pi
    comamand = Serial.read();
    // if it's a capital H (ASCII 72), turn on the LED:
  } else {
    reset(); // not a built-in function
  }

  drive_car(command, time);
}

void reset()
{
    digitalWrite(forward, HIGH);
    digitalWrite(reverse, HIGH);
    digitalWrite(left, HIGH);
    digitalWrite(right, HIGH);
}


void drive_car(int command, int time)
{
  switch (command) {
      case 0: reset(); break;

      // single commands
      case 1: 

      default: Serial.print("Invalid Command");
      // this wont get printed unless its connected to laptop or display
  }
}

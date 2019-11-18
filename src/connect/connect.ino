// source1:
// https://github.com/hamuchiwa/AutoRCCar/blob/master/arduino/rc_keyboard_control.ino

// source2: 
// https://www.instructables.com/id/Hacking-a-RC-Car-With-Arduino-and-Android/

int command;

// there are 6 pins in total
// R - L
// B - F 
// Vd - GND 
// may change to fit schematic
// these are the pin numbers, that the arduino has connected to the car
int forward = 9;    // rear motor - goes forward
int reverse = 10;   // rear motor - goes back
int left    = 11;   // front motor - goes left
int right   = 12;   // front motor - goes right
// dont think our project doesnt have front/rearMTEne

// duration for output
int time = 50;

void setup() {
  // initialize values for forwards, back, left, right
  // pinMode(LED_BUILTIN, OUTPUT);
  pinMode(forward, OUTPUT);
  pinMode(reverse, OUTPUT);
  pinMode(left, OUTPUT);
  pinMode(right, OUTPUT);
  // initialize serial communication
  Serial.begin(9600);
}

void loop() {
  // see if there's incoming serial data: receive command
//  if (Serial.available() > 0) {
//    // read the oldest byte in the serial buffer:
//    // gets command from the laptop, or pi
//    comamand = Serial.read();
//    // if it's a capital H (ASCII 72), turn on the LED:
//  } else {
//    reset(); // not a built-in function
//  }
  command = 1;
  drive_car(command, time);
}

void reset() {
    digitalWrite(forward, HIGH);
    digitalWrite(reverse, HIGH);
    digitalWrite(left, HIGH);
    digitalWrite(right, HIGH);
}

////////// driving functions //////////////////////////
void hit_the_gas(int time) {
    digitalWrite(forward, HIGH);
    digitalWrite(reverse, LOW);

    // when we put in the led lights
//    digitalWrite(ledRed, HIGH);
    delay(time);
}

void go_reverse(int time) {
    digitalWrite(reverse, HIGH);
    digitalWrite(forward, LOW);

    // when we put in the led lights
//    digitalWrite(ledRed, HIGH);
    delay(time);
}

void steer_left(int time) {
    digitalWrite(foward, LOW);
    digitalWrite(left, LOW);
    delay(time);
}

void steer_right(int time) {
    digitalWrite(foward, LOW);
    digitalWrite(right, LOW);
    delay(time);
}
////////// end of driving functions ///////////////////

void drive_car(int command, int time) {
  switch (command) {
      case 0: reset(); break;   // hit the break
      // single commands
      case 1: hit_the_gas(time); break;
      case 2: go_reverse(time); break;

      // combination commands
      case 3: steer_left(time); break;
      case 4: steer_left(time); break;

      default: Serial.print("Invalid Command");
      // this wont get printed unless its connected to laptop or display
  }
}

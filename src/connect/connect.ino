void setup() {
  // put your setup code here, to run once:
  pinMode();
  digitalWrite(13, LOW);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    if (Serial.read() == 's') { // this is what we'll encode in rcbellum.py
      digital.Write(13, HIGH);
      delay(2000);
    }
  } else {
    digitalWrite(13, LOW);  
  }
}

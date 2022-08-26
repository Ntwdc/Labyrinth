#include <Servo.h>
Servo servox;
Servo servoy;
int x = 0;
int y = 0;
unsigned long start_time;
unsigned long stop_time;

void setup() {
  Serial.begin(115200);
  servox.attach(14);
  servoy.attach(12);
}

void loop() {
  if(Serial.available()){
    //start_time = millis();
    String pos = Serial.readStringUntil(';');
    int space = pos.indexOf(" ");
    x = pos.substring(0, space).toInt();
    y = pos.substring(space + 1).toInt();
    servoy.write(y);
    servox.write(x);
    //delay(1);
    //stop_time = millis();
    //Serial.println(stop_time-start_time);
 }
}

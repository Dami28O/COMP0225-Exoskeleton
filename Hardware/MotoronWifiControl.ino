#include <WiFi.h>
#include <Motoron.h>
#include "secrets.h"

char ssid[] = SECRET_SSID;
char pass[] = SECRET_PASS;

int status = WL_IDLE_STATUS;

WiFiServer server(80);

#define EMG_PIN A0
int emgValue = 0;

unsigned long lastEmgRead = 0;
const unsigned long EMG_READ_INTERVAL_MS = 50;

MotoronI2C mc1(1);
MotoronI2C mc2(2);
MotoronI2C mc3(3);
MotoronI2C mc4(4);

int16_t motorSpeed = 0;
const int16_t SPEED_STEP = 100;
const int16_t MAX_SPEED = 800;
const int16_t MIN_SPEED = -800;

unsigned long lastMotorRefresh = 0;
const unsigned long MOTOR_REFRESH_INTERVAL_MS = 200;

enum MotorMode
{
  MODE_DRIVE,
  MODE_BRAKE,
  MODE_COAST
};

MotorMode currentMode = MODE_COAST;

void setupMotoron(MotoronI2C &mc)
{
  mc.reinitialize();
  mc.disableCrc();
  mc.clearResetFlag();
  mc.setCommandTimeoutMilliseconds(2000);
}

int16_t clampSpeed(int value)
{
  if (value > MAX_SPEED) return MAX_SPEED;
  if (value < MIN_SPEED) return MIN_SPEED;
  return value;
}

void applySpeed()
{
  currentMode = MODE_DRIVE;

  mc1.setSpeed(1, motorSpeed);
  mc1.setSpeed(2, motorSpeed);

  mc2.setSpeed(1, motorSpeed);
  mc2.setSpeed(2, motorSpeed);

  mc3.setSpeed(1, motorSpeed);
  mc3.setSpeed(2, motorSpeed);

  mc4.setSpeed(1, motorSpeed);
  mc4.setSpeed(2, motorSpeed);
}

void stopMotors()
{
  motorSpeed = 0;
  currentMode = MODE_BRAKE;

  mc1.setBrakingNow(1, 800);
  mc1.setBrakingNow(2, 800);

  mc2.setBrakingNow(1, 800);
  mc2.setBrakingNow(2, 800);

  mc3.setBrakingNow(1, 800);
  mc3.setBrakingNow(2, 800);

  mc4.setBrakingNow(1, 800);
  mc4.setBrakingNow(2, 800);
}

void coastMotors()
{
  motorSpeed = 0;
  currentMode = MODE_COAST;

  mc1.setSpeed(1, 0);
  mc1.setSpeed(2, 0);

  mc2.setSpeed(1, 0);
  mc2.setSpeed(2, 0);

  mc3.setSpeed(1, 0);
  mc3.setSpeed(2, 0);

  mc4.setSpeed(1, 0);
  mc4.setSpeed(2, 0);

  
}

void sendWebPage(WiFiClient &client)
{
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/html");
  client.println("Connection: close");
  client.println();

  client.println("<!DOCTYPE html>");
  client.println("<html>");
  client.println("<head>");
  client.println("<meta name='viewport' content='width=device-width, initial-scale=1'>");
  client.println("<style>");
  client.println("body{font-family:Arial;text-align:center;margin-top:40px;}");
  client.println("button{font-size:24px;padding:15px 30px;margin:10px;}");
  client.println("input{font-size:24px;width:120px;text-align:center;}");
  client.println("</style>");
  client.println("</head>");

  client.println("<body>");
  client.println("<h1>GIGA Motor Control</h1>");

  client.print("<h2>Current Speed: ");
  client.print(motorSpeed);
  client.println("</h2>");

  client.println("<form action='/set'>");
  client.println("<input type='number' name='speed' min='-800' max='800' value='0'>");
  client.println("<button type='submit'>Set Speed</button>");
  client.println("</form>");

  client.println("<br>");
  client.println("<a href='/inc'><button>Increase</button></a>");
  client.println("<a href='/dec'><button>Decrease</button></a>");
  client.println("<a href='/stop'><button style='background:red;color:white;'>BRAKE STOP</button></a>");
  client.println("<a href='/coast'><button style='background:orange;color:white;'>COAST STOP</button></a>");

  client.println("</body>");
  client.println("</html>");
}

void handleRequest(String request)
{
  if (request.indexOf("GET /inc") >= 0)
  {
    motorSpeed = clampSpeed(motorSpeed + SPEED_STEP);
    applySpeed();
  }
  else if (request.indexOf("GET /dec") >= 0)
  {
    motorSpeed = clampSpeed(motorSpeed - SPEED_STEP);
    applySpeed();
  }
  else if (request.indexOf("GET /stop") >= 0)
  {
    stopMotors();
  }
  else if (request.indexOf("GET /coast") >= 0)
  {
    coastMotors();
  }
  else if (request.indexOf("GET /set?speed=") >= 0)
  {
    int start = request.indexOf("speed=") + 6;
    int end = start;

    while (end < (int)request.length())
    {
      char c = request[end];
      if (c == ' ' || c == '&' || c == '\r' || c == '\n') break;
      end++;
    }

    String speedString = request.substring(start, end);
    int requestedSpeed = speedString.toInt();

    motorSpeed = clampSpeed(requestedSpeed);
    applySpeed();
  }
}

void readEmgIfDue()
{
  if (millis() - lastEmgRead >= EMG_READ_INTERVAL_MS)
  {
    emgValue = analogRead(EMG_PIN);

    Serial.println(emgValue);

    lastEmgRead = millis();
  }
}

void setup()
{
  Serial.begin(115200);
  while (!Serial) {}

  Wire1.begin();

  mc1.setBus(&Wire1);
  mc2.setBus(&Wire1);
  mc3.setBus(&Wire1);
  mc4.setBus(&Wire1);

  setupMotoron(mc1);
  setupMotoron(mc2);
  setupMotoron(mc3);
  setupMotoron(mc4);

  mc1.setMaxAcceleration(1, 400);
  mc1.setMaxDeceleration(1, 400);
  mc1.setMaxAcceleration(2, 400);
  mc1.setMaxDeceleration(2, 400);

  mc2.setMaxAcceleration(1, 400);
  mc2.setMaxDeceleration(1, 400);
  mc2.setMaxAcceleration(2, 400);
  mc2.setMaxDeceleration(2, 400);

  mc3.setMaxAcceleration(1, 400);
  mc3.setMaxDeceleration(1, 400);
  mc3.setMaxAcceleration(2, 400);
  mc3.setMaxDeceleration(2, 400);

  mc4.setMaxAcceleration(1, 400);
  mc4.setMaxDeceleration(1, 400);
  mc4.setMaxAcceleration(2, 400);
  mc4.setMaxDeceleration(2, 400);

  stopMotors();

  pinMode(EMG_PIN, INPUT);
  analogReadResolution(12);
  Serial.println("MyoWare EMG analog read enabled on A0 (12-bit)");

  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  // check for the WiFi module:
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed.");
    while (true);
  }

  const int MAX_WIFI_ATTEMPTS = 5;
  int wifiAttempts = 0;

  
  while (status != WL_CONNECTED && wifiAttempts < MAX_WIFI_ATTEMPTS) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.print(ssid);
    Serial.print(" (attempt ");
    Serial.print(wifiAttempts + 1);
    Serial.print("/");
    Serial.print(MAX_WIFI_ATTEMPTS);
    Serial.println(")");

    status = WiFi.begin(ssid, pass);
    delay(3000);
    wifiAttempts++;
  }
  
  if (status == WL_CONNECTED) {
    printWifiStatus();
    server.begin();
  } else {
    Serial.println("WiFi connection failed — continuing without web control.");
    Serial.println("EMG serial logging and motor control over reset are still active.");
  }
}

void loop()
{
  readEmgIfDue();
  
  if (millis() - lastMotorRefresh >= MOTOR_REFRESH_INTERVAL_MS)
  {
    if (currentMode == MODE_DRIVE)
    {
      applySpeed();
    }
    else if (currentMode == MODE_BRAKE)
    {
      stopMotors();
    }

    lastMotorRefresh = millis();
  }

  if (WiFi.status() == WL_CONNECTED)
  {
    WiFiClient client = server.available();

    if (client)
    {
      String request = "";
      String currentLine = "";
      unsigned long clientTimeout = millis() + 1000;

      while (client.connected() && millis() < clientTimeout)
      {
        if (client.available())
        {
          char c = client.read();

          if (request.length() == 0 && c != '\n' && c != '\r')
          {
          }

          if (c == '\n')
          {
            if (currentLine.length() == 0)
            {
              break;
            }
            if (request.length() == 0)
            {
              request = currentLine;
            }
            currentLine = "";
          }
          else if (c != '\r')
          {
            currentLine += c;
          }
        }
      }

      Serial.println(request);

      handleRequest(request);

      readEmgIfDue();

      sendWebPage(client);

      delay(1);
      client.stop();
    }
  }
}  
/* -------------------------------------------------------------------------- */
void printWifiStatus() {
/* -------------------------------------------------------------------------- */
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your board's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}
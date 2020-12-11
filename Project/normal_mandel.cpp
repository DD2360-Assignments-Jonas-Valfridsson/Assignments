#pragma GCC optimize("O3")
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
using namespace std;
typedef long long ll;

int main() {
  ll current_time = time(nullptr);
  ofstream image (to_string(current_time).append(".bmp"), ofstream::binary);
  image << 
    (uint8_t)0x42 << 
    (uint8_t)0x4D << 
    (uint8_t)0x7C << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x1A << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x0C << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << 
    (uint8_t)0x00 << // Image Width
    (uint8_t)0x4E << // Image Width
    (uint8_t)0xA2 << // Image Height
    (uint8_t)0x45 << // Image height
    (uint8_t)0x01 << 
    (uint8_t)0x00 << 
    (uint8_t)0x18 << 
    (uint8_t)0x00;

  char start1 = 200, start2 = 200, start3 = 200;
  int max_iter = 150, shade = 1, speed1 = 0, speed2 = 10, speed3 = 0, j = 1;
  char colors[max_iter * 3];
  for (int i = 0; i < max_iter; i+=3) {
    if (j % 50 == 0)
      shade <<= 1;

    int red = start1 + i * speed1 - j;
    int green = start2 + i * speed2;
    int blue = start3 + i * speed3 - j;

    if (red < 0) red = 0;
    if (green < 0) green = 0;
    if (blue < 0) blue = 0;


    colors[i] =     (red) % (256 / shade);
    colors[i + 1] = (green) % (256 / shade);
    colors[i + 2] = (blue) % (256 / shade);

    j += 1;
  }


  int xpixels = 19968, ypixels = 13730;
  char *memory = (char*)malloc(sizeof(char) * xpixels * ypixels * 3);

  auto start = std::chrono::steady_clock::now();
  for (double y = 0; y < ypixels; y++) {
    for(double x = 0; x < xpixels; x++) {
      double c_re = (x - xpixels/2.0)*4.0/xpixels;
      double c_im = (y - ypixels/2.0)*4.0/xpixels;
      double i = 0, j = 0;
      int iteration = 0;
      while ( i*i + j*j < 4 && iteration < max_iter) {
        double i_new = i*i - j*j + c_re;
        j = 2*i*j + c_im;
        i = i_new;
        iteration++;
      }	
      int coordinate = (x + y * xpixels) * 3;
      if (iteration < max_iter) {
        memory[coordinate] = colors[3*iteration];
        memory[coordinate + 1] = colors[3*iteration + 1];
        memory[coordinate + 2] = colors[3*iteration + 2];

      } else {
        memory[coordinate] = start1;
        memory[coordinate + 1] = start2;
        memory[coordinate + 2] = start3;
      }
    }	
  }

  auto end = std::chrono::steady_clock::now();
  std::cout << "RUN " 
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
    << std::endl;

  for (int i = 0; i < xpixels * ypixels * 3; i++)
    image << memory[i];

  image << 0x00 << 0x00;	

  free(memory);
  return 0;
}

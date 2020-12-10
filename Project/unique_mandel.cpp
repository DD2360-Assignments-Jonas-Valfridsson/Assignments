#pragma GCC optimize("O3")
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>
using namespace std;
typedef long long ll;

int main() {
	ll current_time = time(nullptr);
	ofstream image (to_string(current_time).append(".bmp"), ofstream::binary);
	image << (uint8_t)0x42 << (uint8_t)0x4D << (uint8_t)0x7C << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x1A << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x0C << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x00 << (uint8_t)0x0A << (uint8_t)0xA0 << (uint8_t)0x05 << (uint8_t)0x01 << (uint8_t)0x00 << (uint8_t)0x18 << (uint8_t)0x00;
	
	uint8_t start1 = current_time%256, start2 = ((current_time%1000000 - current_time%1000)/1000)%256, start3 = ((current_time%1000000000 - current_time%1000000)/1000000)%256;
	int start1_iterative_speed = current_time%10, start2_iterative_speed = current_time%10000 - current_time%1000, start3_iterative_speed = current_time%1000000 - current_time%100000;
	
	int max_iter = 1000, shade = 1;
	uint8_t *colors = new uint8_t[max_iter * 3];
	for (int i = 0; i < max_iter; i+=3) {
		if (i%150 == 0)
			shade <<= 1;

		colors[i] =  (start1 + i*start1_iterative_speed)%(256/shade);
		colors[i+1] = (start2 + i*start2_iterative_speed)%(256/shade);
		colors[i+2] = (start3 + i*start3_iterative_speed)%(256/shade);
	}

	for (double y = 0; y < 1440; y++) {
		for(double x = 0; x < 2560; x++) {
			double c_re = (x - 2560/2.0)*4.0/2560;
			double c_im = (y - 1440/2.0)*4.0/2560;
			double i = 0, j = 0;
			int iteration = 0;
			while ( i*i + j*j < 4 && iteration < max_iter) {
				double i_new = i*i - j*j + c_re;
				j = 2*i*j + c_im;
				i = i_new;
				iteration++;
			}	
			if (iteration < max_iter) {
				image << colors[3*iteration] << colors[3*iteration+1]  << colors[3*iteration+2];
        printf("%d %d %d\n", colors[3 * iteration], colors[3*iteration+1], colors[3*iteration+2]);
			} else {
				image << start1 << start2 << start3;
			}
		}	
	}

	image << 0x00 << 0x00;	
	delete[] colors;
	return 0;

}

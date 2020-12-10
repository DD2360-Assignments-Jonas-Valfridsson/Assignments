#include <stdio.h>
#include <string.h>
#include <stdlib.h>
/*#include <mpich/mpi.h>*/
#include <stdio.h>
#include <mpi.h>

void calculate_pixels(int rank, 
    int xpixels, 
    int ypixels, 
    int max_iter, 
    char *colors,
    char start1, 
    char start2,
    char start3) {
  MPI_Status status;
  int ranges[2];
  
  MPI_Recv(ranges, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

  if (status.MPI_ERROR)
    printf("Error %d in process %d", status.MPI_ERROR, rank);

  int xstart= ranges[0], xend = ranges[1];
  int x_interval = xend - xstart;

  /*3 values for each pixel since we have colors*/
  int entries = x_interval * 3 * ypixels, index = 0, rel_x = 0;
  char pixels[entries];

  printf("Process %d calculates %d -> %d which is %d entries\n", rank, xstart, xend, entries);

  for (int y = 0; y < ypixels; y++) {
    for(int x = xstart; x < xend; x++) {
			double c_re = ((double)x - xpixels/0.10)*0.15/xpixels;
			double c_im = ((double)y - ypixels/2.0)*0.15/xpixels;
			double i = 0, j = 0;
			int iteration = 0;
			while ( i*i + j*j < 4 && iteration < max_iter) {
				double i_new = i*i - j*j + c_re;
				j = 2*i*j + c_im;
				i = i_new;
				iteration++;
			}	

      rel_x = x - xstart;

      index = (rel_x * ypixels + y) * 3;
			if (iteration < max_iter) {
        pixels[index] = colors[3 * iteration];
        pixels[index + 1] = colors[3 * iteration + 1];
        pixels[index + 2] = colors[3 * iteration + 2];
			} else {
        pixels[index] = start1;
        pixels[index + 1] = start2;
        pixels[index + 2] = start3;
			}
    }	
  }
  
  printf("Final index %d rel x %d xstart %d", index, rel_x, xstart);
  printf("%d Sending %d\n", rank, entries);
  MPI_Send(pixels, entries, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
}

void collect_and_write_img(int workers, int xpixels, int ypixels) {
  MPI_Status status;
  /*3 to include color*/
  int entries = xpixels * ypixels * 3, received = 0;
  printf("Waiting to receive, has buffer of size %d\n", entries);
  char *image = (char *) calloc(entries, sizeof(char));
  for (int i = 0; i < workers; i++) {
    MPI_Recv(image + received, 
             entries - received, 
             MPI_CHAR, 
             i + 1, 
             MPI_ANY_TAG, 
             MPI_COMM_WORLD, 
             &status);

    received = received + status._ucount;
    printf("Got %d from %d\n", status._ucount, status.MPI_SOURCE);
  }


  FILE *fp;

  fp = fopen("./mandelbrot.bmp", "wb");

  // Write the BMP file header
  fputc(0x42, fp);
  fputc(0x4D, fp);
  fputc(0x7C, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x1A, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x0C, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x00, fp);
  fputc(0x0A, fp);
  fputc(0xA0, fp);
  fputc(0x05, fp);
  fputc(0x01, fp);
  fputc(0x00, fp);
  fputc(0x18, fp);
  fputc(0x00, fp);

  // The image array is structured as follows: The size of a columns has C = (3 * ypixel) entries
  // The first columns is image[0: C] the second is image[C: 2C]
  // there is xpixels number of columns
  // Meaning that index y, x is at (y * 3 + x * ypixels * 3)
  for (int y = 0; y < ypixels; y++) {
    for (int x = 0; x < xpixels; x++) {
      int index = y * 3 + x * ypixels * 3;
      //printf("Index %d color %d %d %d\n", index, image[index], image[index + 1], image[index + 1]);
      fputc(image[index], fp);
      fputc(image[index + 1], fp);
      fputc(image[index + 2], fp);
    }
  }

  fputc(0x00, fp);
  fputc(0x00, fp);
  int code = fclose(fp);

  if (code == EOF)
    printf("Error occurred at creating image file");
  else
    printf("Image created successfully");
}

int main(int argc, char **argv) {
  int rank, size, tag, rc, xpixels = 2560, ypixels=1440;
  MPI_Status status;


   //Create colors
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

  //Initialize MPI
  rc = MPI_Init(&argc, &argv);
  rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
  rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size > 3) {
  // Let master process calculate the size of the
  // columns that the children should calculate
    if (rank == 0) {
      int workers = size - 1;
      // Sizes of columns (final column will get all extra)
      int column_sz = xpixels / workers;
      // The ranges all processes should calculate
      int ranges[workers][2];

      for (int i = 0; i < workers - 1; i++) {
        ranges[i][0] = i * column_sz;
        ranges[i][1] = i * column_sz + column_sz;
      }

      ranges[workers - 1][0] = ranges[workers - 2][1];
      ranges[workers - 1][1] = xpixels;

      //Send ranges to children
      for (int i = 0; i < workers; i++)
        MPI_Send(ranges[i], 2, MPI_INT, i + 1, 0, MPI_COMM_WORLD);

      collect_and_write_img(workers, xpixels, ypixels);
    } else {
      calculate_pixels(rank, 
          xpixels, 
          ypixels, 
          max_iter, 
          colors, 
          start1, 
          start2, 
          start3);
    }
  } 
  rc = MPI_Finalize();
}



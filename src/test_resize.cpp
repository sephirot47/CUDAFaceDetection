#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stbi.h"
#include "stbi_write.h"

typedef unsigned char uc;
using namespace std;

class Pixel {
public:
  int x,y;
  Pixel() { x = y = 0; }
  Pixel(int _x, int _y) {x=_x; y=_y;}
  string toString() {
    ostringstream ss;
    ss << "(" << x << ",\t" << y << ")";
    return ss.str();
  }
};

class Color {
public:
  uc r,g,b,a;
  Color(uc _r, uc _g, uc _b, uc _a) { r=_r; g=_g; b=_b; a=_a; }
  string toString() {
    ostringstream ss;
    ss << "(" << r << "," << g << "," << b << "," << a << ")";
    return ss.str();
  }
};

class Image
{
public:
  char* filename;

  Image(char* filename) {
    this->filename = filename;
    FILE *file = fopen(filename, "r");
    if (file != NULL) {
        data = stbi_load_from_file(file, &_width, &_height, &comp, 4); // rgba
        fclose (file);
        printf("%s read successfully\n", filename);
    }
    else printf("Image '%s' not found\n", filename);
  }

  int width() { return _width;  }
  int height() { return _height; }

  Color getColor(Pixel p) {
      int offset = (p.y * _width + p.x) * comp;
      return Color((data[offset + 0]), (data[offset + 1]),
                   (data[offset + 2]), (data[offset + 3]));
  }

  uc getGrayScale(Pixel p) { return grayscale(getColor(p)); }

private:
  uc *data;
  int _width, _height, comp;
  uc grayscale(const Color &c) { return 0.299*c.r + 0.587*c.g + 0.114*c.b; }
};


class FaceDetection
{
public:
  Image *container;

  FaceDetection(char* containerFile)
  {
    container = new Image(containerFile);
  }

  // 0:   same grayscale
  // 255: very different
  uc getDiff(uc gray1, uc gray2)
  {
    return abs(gray1-gray2);
  }

  /*
  void saveResult(Pixel origin) {
      uc *result = new uc[container->width() * container->height() * 3 * sizeof(uc)];
      for(int y = 0; y < container->height(); ++y) {
          for(int x = 0; x < container->width();  ++x) {
              Pixel p(x,y);
              int offset = (y * container->width() + x) * 3;
              if (belongsTo(p,origin,object->width(),object->height())) {
                  result[offset + 0] = container->getColor(p).r;
                  result[offset + 1] = 255;
                  result[offset + 2] = container->getColor(p).b;
              }
              else {
                  result[offset + 0] = container->getColor(p).r;
                  result[offset + 1] = container->getColor(p).g;
                  result[offset + 2] = container->getColor(p).b;
              }
          }
      }
      stbi_write_bmp("result.bmp", container->width(), container->height(), 3, result);
  }
  */

private:
  bool belongsTo(Pixel p, Pixel o, int w, int h) {
      return p.x >= o.x && p.x < o.x+w && p.y >= o.y && p.y < o.y+h;
  }

};

void saveImage(uc *img, int x, int y, int width, int height, int imgWidth, const char *filename)
{
    uc *aux = (uc*) malloc(width * height * 3 * sizeof(uc));
    for(int iy = 0; iy < height; ++iy)
    {
        for(int ix = 0; ix < width; ++ix)
        {
            int offset = (iy * width + ix) * 3;
            aux[offset + 0] = img[(iy+y) * imgWidth + ix + x];
            aux[offset + 1] = img[(iy+y) * imgWidth + ix + x];
            aux[offset + 2] = img[(iy+y) * imgWidth + ix + x];
        }
    }
    stbi_write_bmp(filename, width, height, 3, aux);
    free(aux);
}

//Increase contrast
void histogramEqualization(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
    int intensityToNPixels[256];
    for(int i = 0; i < 256; ++i) intensityToNPixels[i] = 0;

    for(int y = oy; y < oy + height; ++y)
    {
        for(int x = ox; x < ox + width; ++x)
        {
            int offset = y * imgWidth + x;
            uc v = img[offset];
            intensityToNPixels[v]++;
        }
    }

    int npixels = width * height;
    float accumulatedProbs[256];
    accumulatedProbs[0] = intensityToNPixels[0];
    for(int i = 1; i < 256; ++i) accumulatedProbs[i] = accumulatedProbs[i-1] + intensityToNPixels[i];

    for(int y = oy; y < oy + height; ++y)
    {
        for(int x = ox; x < ox + width; ++x)
        {
            fflush(stdout);
            int offset = y * imgWidth + x;
            uc v = img[offset];
            img[offset] = floor( 255 * (accumulatedProbs[v] / npixels) );
        }
    }
}

//Always to a smaller size
void resize(uc *src, int srcx, int srcy, int srcw, int srch, int srcTotalWidth, //x,y,width,height
            uc *dst, int dstx, int dsty, int dstw, int dsth, int dstTotalWidth) //x,y,width,height
{
    //Every square of size (bw,bh), will be substituted
    //by one pixel in the dst image
    int bw = srcw / dstw;
    int bh = srch / dsth;

    //For each pixel in the dst
    for(int dy = dsty; dy < dsty + dsth; ++dy)
    {
        for(int dx = dstx; dx < dstx + dstw; ++dx)
        {
            //Offset per dst pixel. Every pixel we move in x,y in dst,
            //we move bw,bh in the src image.
            int resizeOffset = (dy * bh) * srcTotalWidth + (dx * bw);

            //Save in its position the mean of the corresponding window pixels
            int mean = 0;
            for(int sy = 0; sy < bh; ++sy)
            {
                for(int sx = 0; sx < bw; ++sx)
               {
                    int srcOffset = (srcy + sy) * srcTotalWidth + (srcx + sx);
                    uc v = src[srcOffset + resizeOffset];
                    mean += v;
                }
            }
            mean /= bw * bh;
            dst[dy * dstTotalWidth + dx] = mean;
        }
    }
}

uc getHeuristic9x9(uc *img) {
    int v = int(img[22]) - int( (int(img[19])+int(img[20])+int(img[24])+int(img[25]))/4);
    return v < 0 ? 0 : v;
}

int main(int argc, char** argv)
{
  FaceDetection fc(argv[1]);

  int numBytesContainer = fc.container->width() * fc.container->height() * sizeof(uc);

  printf("Container File: %s, size(%d px, %d px), bytes(%d B)\n",
         fc.container->filename, fc.container->width(), fc.container->height(),
         numBytesContainer);

  // Obtener Memoria en el host
  uc *h_containerImageGS = (uc*) malloc(numBytesContainer);
  printf("Filling ContainerGS in the host with GS values...\n");
  for(int y = 0; y < fc.container->height(); ++y) {
      for(int x = 0; x < fc.container->width(); ++x) {
          h_containerImageGS[y * fc.container->width() + x] = fc.container->getGrayScale(Pixel(x,y));
      }
  }

  resize(h_containerImageGS, 1300, 1200, 550, 550, fc.container->width(),
         h_containerImageGS, 0, 0, 9, 9, fc.container->width());
  histogramEqualization(h_containerImageGS, 0, 0, 9, 9, fc.container->width());
  saveImage(h_containerImageGS, 0, 0, 9, 9, fc.container->width(), "test.bmp");

  int windowWidth = 550;
  int windowHeight = 550;
  int step = 50;
  for (int y = 0; y < fc.container->height() - windowHeight; y += step)
      for (int x = 0; x < fc.container->width() - windowWidth; x += step) {
          uc window9x9[81];
          resize(h_containerImageGS, x, y, windowWidth, windowHeight, fc.container->width(),
                 window9x9, 0, 0, 9, 9, 9);
          histogramEqualization(window9x9, 0, 0, 9, 9, 9);
          uc hv = getHeuristic9x9(window9x9);
          printf("%d ... H(%d,%d) -> %d\n", hv, x, y, hv);
          if (hv == 255) {
              printf("\tSaving window...\n");
              string filename = "output/window";
              filename += to_string(x); filename += to_string(y);
              filename += ".bmp";
              saveImage(window9x9, 0, 0, 9, 9, 9, filename.c_str());
          }
      }

  printf("Image saved!");
}








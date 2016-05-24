#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
#define IMG_CHANNELS 3

//Optimal values 40, 550
#define THRESH_9x9 40     //Bigger = more restrictive
#define THRESH_30x30 700  //Bigger = less restrictive

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
  int r,g,b,a;
  Color(int _r, int _g, int _b, int _a) { r=_r; g=_g; b=_b; a=_a; }
  string toString() {
    ostringstream ss;
    ss << "(" << r << "," << g << "," << b << "," << a << ")";
    return ss.str();
  }
};

class Image
{
public:
  uc *data;
  char* filename;

  Image(char* filename) {
    this->filename = filename;
    FILE *file = fopen(filename, "r");
    if (file != NULL) {
        data = stbi_load_from_file(file, &_width, &_height, &comp, IMG_CHANNELS); // rgb
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
  int _width, _height, comp;
  uc grayscale(const Color &c) { return 0.299*c.r + 0.587*c.g + 0.114*c.b; }
};

struct Box { int x,y,w,h; Box(int _x, int _y, int _w, int _h) { x=_x; y=_y; w=_w; h=_h; } };

class FaceDetection
{
public:
  vector<Box> resultWindows;
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

  void saveResult()
  {
      int boxStroke = 1;
      printf("Saving result...\n");

      uc *result = new uc[container->width() * container->height() * 3 * sizeof(uc)];
      for(int y = 0; y < container->height(); ++y)
      {
          for(int x = 0; x < container->width();  ++x)
          {
              Pixel p(x,y);
              bool isBoundary = false;
              for(Box b : resultWindows)
              {
                  if(belongsTo(p, b))
                  {
                      if(abs(x - b.x) <= boxStroke ||
                         abs(y - b.y) <= boxStroke ||
                         abs(x - (b.x + b.w - 1)) <= boxStroke ||
                         abs(y - (b.y + b.h - 1)) <= boxStroke)
                      {
                          isBoundary = true;
                      }
                  }
              }

              int offset = (y * container->width() + x) * 3;
              if(isBoundary)
              {
                  result[offset + 0] = 255 - container->getColor(p).r;
                  result[offset + 1] = 255 - container->getColor(p).g;
                  result[offset + 2] = 255 - container->getColor(p).b;
              }
              else
              {
                  result[offset + 0] = container->getColor(p).r;
                  result[offset + 1] = container->getColor(p).g;
                  result[offset + 2] = container->getColor(p).b;
              }
          }
      }

      stbi_write_bmp("output/result.bmp", container->width(), container->height(), 3, result);
      //saveImage(result, 0, 0, container->width(), container->height(), container->width()*3, );
  }

private:
  bool belongsTo(Pixel p, Box b) {
      return p.x >= b.x && p.x <= b.x+b.w && p.y >= b.y && p.y <= b.y+b.h;
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

// Resize always to a smaller size -> downsample
void resize(uc *src, int srcx, int srcy, int srcw, int srch, int srcTotalWidth, //x,y,width,height
            uc *dst, int dstx, int dsty, int dstw, int dsth, int dstTotalWidth) //x,y,width,height
{
    //Every square of size (bw,bh), will be substituted
    //by one pixel in the dst image
    float bw = float(srcw) / dstw;
    float bh = float(srch) / dsth;

    //For each pixel in the dst
    for(int dy = dsty; dy < dsty + dsth; ++dy)
    {
        for(int dx = dstx; dx < dstx + dstw; ++dx)
        {
            //Offset per dst pixel. Every pixel we move in x,y in dst,
            //we move bw,bh in the src image.
            int resizeOffset = ceil((dy * bh)) * srcTotalWidth + ceil((dx * bw));

            //Save in its position the mean of the corresponding window pixels
            int mean = 0;
            for(int sy = 0; sy < floor(bh); ++sy)
            {
                for(int sx = 0; sx < floor(bw); ++sx)
               {
                    int srcOffset = (srcy + sy) * srcTotalWidth + (srcx + sx);
                    uc v = src[srcOffset + resizeOffset];
                    mean += v;
                }
            }

            mean /= floor(bw * bh);
            dst[dy * dstTotalWidth + dx] = mean;
        }
    }
}

float getHistogram(uc *img, int ox, int oy, int width, int height, int imgWidth, float histogram[256]) {
  
  float npixels = width * height;
  float unitProb = 1.0f/npixels;
  float maxfreq = 0.0f;
  for(int i = 0; i < 256; ++i) histogram[i] = 0;
  for(int y = oy; y < oy + height; ++y)
  {
      for(int x = ox; x < ox + width; ++x)
      {
	  int offset = y * imgWidth + x;
	  uc v = img[offset];
	  histogram[v] += unitProb;
      if(histogram[v] > maxfreq) maxfreq = histogram[v];
      }
  }
  return maxfreq;
}

float histogramHeuristic(float histogram[256], float maxFreq)
{
    //Worst case: 5.4;
    float h = 0;
    h += (histogram[10] / maxFreq);
    h += (histogram[75] / maxFreq);
    h += 3*fabs(0.8f - (histogram[190] / maxFreq));
    h += (histogram[240] / maxFreq);
    return h;
}

void plotHistogram(float histogram[256], float maxv, const char *filename)
{
    int width = 256;
    int height = 110;
    uc *aux = (uc*) malloc(width * height * 3 * sizeof(uc));
    
    for(int iy = 0; iy < height; ++iy)
    {
        for(int ix = 0; ix < width; ++ix)
        {
	    int offset = (iy * width + ix) * 3;
	    if (height-iy-1 == uc(histogram[ix] * (height / maxv))) {
	      aux[offset + 0] = 0;
	      aux[offset + 1] = 0;
	      aux[offset + 2] = 0;
	    }
	    else {
	      aux[offset + 0] = 255;
	      aux[offset + 1] = 255;
	      aux[offset + 2] = 255;
	    }
        }
    }
    // for(int i = 0; i < 256; ++i) printf("%d: %f\n",i,histogram[i]);
    stbi_write_bmp(filename, width, height, 3, aux);
    free(aux);
}

void toBlackAndWhite(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
    for(int y = oy; y < oy + height; ++y)
    {
        for(int x = ox; x < ox + width; ++x)
        {
            int offset = y * imgWidth + x;
            uc v = img[offset];
            img[offset] = v > 200 ? 255 : 0;
        }
    }
}

// Increase contrast
void histogramEqualization(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
    float histogram[256];
    getHistogram(img, ox, oy, width, height, imgWidth, histogram);
  
    float accumulatedProbs[256];
    accumulatedProbs[0] = histogram[0];
    for(int i = 1; i < 256; ++i) accumulatedProbs[i] = accumulatedProbs[i-1] + histogram[i];

    for(int y = oy; y < oy + height; ++y)
    {
        for(int x = ox; x < ox + width; ++x)
        {
            int offset = y * imgWidth + x;
            uc v = img[offset];
            img[offset] = floor(255 * accumulatedProbs[v]);
        }
    }
}

uc getFirstStageHeuristic(uc *img) {
    int v = img[22] - (img[19]+img[20]+img[24]+img[25]+img[58])/5;
    return v < 0 ? 0 : v;
}

// Find edges in horizontal direction
void sobelEdgeDetection(uc *img, int ox, int oy, int winWidth, int winHeight, int imgWidth, uc *&sobelImg)
{
    uc threshold = 24;
    
    for(int y = oy; y < oy + winHeight; ++y)
    {
        for(int x = ox; x < ox + winWidth; ++x)
        {
            int imgOffset = y * imgWidth + x;
            int winOffset = (y-oy) * winWidth + (x-ox);

            if (y == oy or y == oy+winHeight-1 or x == ox or x == ox+winWidth-1)
                sobelImg[winOffset] = 255;
            else {
                uc upperLeft  = img[imgOffset - imgWidth - 1];
                uc upperRight = img[imgOffset - imgWidth + 1];
                uc up         = img[imgOffset - imgWidth];
                uc down       = img[imgOffset + imgWidth];
                uc left       = img[imgOffset - 1];
                uc right      = img[imgOffset + 1];
                uc lowerLeft  = img[imgOffset + imgWidth - 1];
                uc lowerRight = img[imgOffset + imgWidth + 1];

                //int sum = -upperLeft + upperRight + -2*left + 2*right + -lowerLeft + lowerRight;
                int sum = -upperLeft - upperRight - 2*up + 2*down + lowerLeft + lowerRight;
                //int sum = -left + right;

                if(sum >= threshold)
                    sobelImg[winOffset] = 0;
                else
                    sobelImg[winOffset] = 255;
            }
        }
    }
}

uc getWindowMeanGS(uc *img, int ox, int oy, int winWidth, int winHeight, int imgWidth) {
    int sum = 0;
    for(int y = oy; y < oy + winHeight; ++y)
    {
        for(int x = ox; x < ox + winWidth; ++x)
        {
            int offset = y * imgWidth + x;
            sum += img[offset];
        }
    }
    
    return uc(sum / (winWidth * winHeight));
}

Color getWindowMeanColor(uc *img, int ox, int oy, int winWidth, int winHeight, int imgWidth) {

    Color csum(0,0,0,0);
    for(int y = oy; y < oy + winHeight; ++y)
    {
        for(int x = ox; x < ox + winWidth; ++x)
        {
            int offset = (y * imgWidth + x) * 3;
            csum.r += img[offset + 0];
            csum.g += img[offset + 1];
            csum.b += img[offset + 2];
        }
    }

    int npixels = winWidth * winHeight;
    Color c(csum.r / npixels, csum.g / npixels, csum.b / npixels, 0);

    return c;
}

int getSecondStageHeuristic(uc *img) {
    int sumDiff = 0;

    int leftEye = getWindowMeanGS(img,2,4,9,5,30);
    sumDiff += leftEye;

    int rightEye = getWindowMeanGS(img,18,4,9,5,30);
    sumDiff += rightEye;

    //simmetry
    sumDiff += abs(rightEye - leftEye);

    // upper nose
    sumDiff += 255-getWindowMeanGS(img,11,1,6,13,30);

    // lower nose
    sumDiff += abs(125 - getWindowMeanGS(img,10,15,9,5,30));

    // left cheek
    int leftCheek = 255-getWindowMeanGS(img,1,10,8,10,30);
    sumDiff += leftCheek;

    // right cheek
    int rightCheek = 255-getWindowMeanGS(img,19,10,8,10,30);
    sumDiff += rightCheek;

    sumDiff += abs(leftCheek - rightCheek);

    // mouth
    sumDiff += getWindowMeanGS(img,8,21,13,5,30);

    return sumDiff;
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

  printf("Resizing original image....\n");
  int numBytesImage = IMG_WIDTH * IMG_WIDTH * sizeof(uc);
  uc *h_imageGS = (uc*) malloc(numBytesImage);
  resize(h_containerImageGS,
         0, 0, fc.container->width(), fc.container->height(), fc.container->width(),
         h_imageGS,
         0, 0, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH
         );
  saveImage(h_imageGS,0,0,IMG_WIDTH,IMG_HEIGHT,IMG_WIDTH,"test.bmp");
  //

  const int thresh1 = 40; //higher = more restrictive
  const float histoThresh = 99.9f;//3.0f; //higher = less restrictive (0.0f->3.8f)
  const int thresh2 = 550; //higher = less restrictive

  //Original 100,100
  int winWidth = 34;
  int winHeight = 51;
  float widthRatio = float(fc.container->width()) / IMG_WIDTH;
  float heightRatio = float(fc.container->height()) / IMG_HEIGHT;
  int step = 15;
  for (int y = 0; y < IMG_HEIGHT - winHeight; y += step)
  {
      for (int x = 0; x < IMG_WIDTH - winWidth; x += step)
      {
          string filename = "output/win_" + to_string(x) + "_" + to_string(y);

          // FIRST STAGE
          // resize window to 9x9
          uc window9x9[9*9];
          resize(h_imageGS, x, y, winWidth, winHeight, IMG_WIDTH,
                 window9x9, 0, 0, 9, 9, 9);
          histogramEqualization(window9x9, 0, 0, 9, 9, 9);

          uc hv1 = getFirstStageHeuristic(window9x9);
          if (hv1 >= THRESH_9x9)
          {
              // SECOND STAGE
                  // apply sobel edge detection
                  uc *sobelImg = (uc*) malloc(winWidth * winHeight * sizeof(uc));
                  sobelEdgeDetection(h_imageGS, x, y, winWidth, winHeight, IMG_WIDTH, sobelImg);

                  // resize window to 30x30
                  uc window30x30[30*30];
                  resize(sobelImg, 0, 0, winWidth, winHeight, winWidth,
                         window30x30, 0, 0, 30, 30, 30);
                  toBlackAndWhite(window30x30, 0, 0, 30, 30, 30);

                  free(sobelImg);
                  int hv2 = getSecondStageHeuristic(window30x30);
                  if (hv2 <= THRESH_30x30)
                  {
                      //saveImage(window30x30, 0, 0, 30, 30, 30, filename.c_str());
                      fc.resultWindows.push_back( Box(int(x * widthRatio),
                                                      int(y * heightRatio),
                                                      int(winWidth * widthRatio),
                                                      int(winHeight * heightRatio)));
                  }
           }
      }
  }

  fc.saveResult();
  printf("Result saved to 'output/result.bmp'\n");
  //system("xdg-open output/result.bmp");
}








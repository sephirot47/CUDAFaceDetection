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

class Pixel
{
public:
  int x,y; 

  Pixel(int _x, int _y) {x=_x; y=_y;}

  string toString() { 
    ostringstream ss;
    ss << "(" << x << ",\t" << y << ")";
    return ss.str();
    
  }
};

class Color
{
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
  Image(char* filename) {
    FILE *file = fopen(filename, "r");
    if (file != NULL)
    {
      data = stbi_load_from_file(file, &_width, &_height, &comp, 4); // rgba
      fclose (file);
    }
  }

  int width() {
      return _width;
  }

  int height() {
      return _height;
  }
  
  Color getColor(Pixel p) {
      int offset = (p.y * _width + p.x) * comp;
      Color c((data[offset + 0]),
              (data[offset + 1]),
              (data[offset + 2]),
              (data[offset + 3]));
      return c;
  }
  
  uc getGrayScale(Pixel p) {
    return grayscale(getColor(p));
  }

private:
  uc *data = 0;
  int _width, _height, comp;

  uc grayscale(const Color &c) {
      return 0.299*c.r + 0.587*c.g + 0.114*c.b;
  }
  
};


class FaceDetection 
{
public:
  Image *container, *object;
  
  FaceDetection(char* containerFile, char* objectFile)
  {
    container = new Image(containerFile);
    object = new Image(objectFile);
  }
  
  // 0:   exact same image
  // 255: very different
  uc getWindowDiff(Pixel origin, int windowWidth, int windowHeight)
  {
    int totalDiff = 0;
    for(int y = 0; y < windowHeight; ++y)
    {
      for(int x = 0; x < windowWidth; ++x)
      {
        Pixel cPixel(origin.x + x, origin.y + y); // container pixel
        Pixel oPixel(x,y);                        // object pixel

        uc cAlpha = container->getColor(cPixel).a;
        uc oAlpha = container->getColor(oPixel).a;
        uc cGray = container->getGrayScale(cPixel);
        uc oGray = object->getGrayScale(oPixel);

        uc diffAlpha = getDiff(cAlpha, oAlpha);
        if(diffAlpha < 200) // if alpha channels differ, ignore the diff (consider pixels are equal)
        {
          uc diff = getDiff(cGray, oGray);
          totalDiff += diff;
        }
      }
    }
    
    return uc(totalDiff / (windowWidth * windowHeight));
  }
  
    
  // 0:   same grayscale
  // 255: very different
  uc getDiff(uc gray1, uc gray2) 
  {
    return abs(gray1-gray2);
  }

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

private:
  bool belongsTo(Pixel p, Pixel o, int w, int h) {
      return p.x >= o.x && p.x < o.x+w && p.y >= o.y && p.y < o.y+h;
  }
  
};

int main(int argc, char **argv)
{    
    if(argc < 3)
        cout << "usage: " << argv[0] << " <container file name> <object file name>" << endl;
    
    FaceDetection fc(argv[1], argv[2]);

    int minDiff = 255;
    Pixel minP = Pixel(0,0);
    for(int y = 0; y < fc.container->height() - fc.object->height(); y+=50)//++y)
    {
        for(int x = 0; x < fc.container->width() - fc.object->width();  x+=50)//++x)
        {
            Pixel p(x,y);
            uc diff = fc.getWindowDiff(p, fc.object->width(), fc.object->height());
            cout << p.toString() << endl;
            if (diff < minDiff) { minDiff = diff; minP = p; }
        }
    }
    cout << endl << "Best guess: " << minP.toString() << endl;
    fc.saveResult(minP);
}

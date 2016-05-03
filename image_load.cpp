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
    ostringstream oss;
    oss << "(" << x << ",\t" << y << ")";
    return oss.str();
    
  }
};

class Color
{
public:
  uc r,g,b,a;
  Color(uc _r, uc _g, uc _b, uc _a) { r=_r; g=_g; b=_b; a=_a; }
  string toString() { 
    ostringstream oss;
    oss << "(" << r << "," << g << "," << b << "," << a << ")";
    return oss.str();
  }
};

class Image 
{  
private:
  uc grayscale(const Color &c) {
      return 0.299*c.r + 0.587*c.g + 0.114*c.b;
  }
  
public:
  uc *data = 0;
  int width, height, numComponents;
  
  
  Image(char* filename) {
    FILE * file = fopen(filename, "r");
    if (file != NULL)
    {
      data = stbi_load_from_file(file, &width, &height, &numComponents, 4);
      fclose (file);
    }
  }
  
  
  Color getColor(Pixel p) {
      int offset = (p.y * width + p.x) * numComponents;
      Color c(((int)data[offset + 0]),
	      ((int)data[offset + 1]),
	      ((int)data[offset + 2]),
	      ((int)data[offset + 3]));
      return c;
  }
  
  uc getGrayScale(Pixel p) {
    return grayscale(getColor(p));
  }
  
};


class FaceDetection 
{
public:
  Image *container, *object;
  
  FaceDetection(char* container_f, char* object_f) 
  {
    container = new Image(container_f);
    object = new Image(object_f);
  }
  
  //0:   exact same image.
  //255: very different. 
  uc getWindowDiff(Pixel origin, int windowWidth, int windowHeight)
  {
    int totalDiff = 0;
    for(int y = 0; y < windowHeight; ++y)
    {
      for(int x = 0; x < windowWidth; ++x)
      {
	Pixel cPixel(origin.x + x,
		     origin.y + y);
	Pixel oPixel(x,y);
	
	uc cAlpha = container->getColor(cPixel).a;
	uc oAlpha = container->getColor(oPixel).a;
	uc cGray = container->getGrayScale(cPixel);
	uc oGray = object->getGrayScale(oPixel);
	
	uc diffAlpha = getDiff(cAlpha, oAlpha);
	if(diffAlpha < 200) 
	{
	  uc diff = getDiff(cGray, oGray);
	  totalDiff += diff;
	}
      }
    }
    
    return uc(totalDiff / (windowWidth * windowHeight));
  }
  
    
  //0:   same grayscale.
  //255: very different. 
  uc getDiff(uc gray1, uc gray2) 
  {
    return abs(gray1-gray2);
  }
  

};

int main(int argc, char **argv)
{    
    if(argc < 3)
    {
      cout << "Pass as arguments container file name, and the object file name." << endl;
    }
    
    FaceDetection fc(argv[1], argv[2]);

    //uc *grayData = new uc[width * height * 3 * sizeof(uc)];
    for(int y = 0; y < fc.container->height - fc.object->height; y+=50)//++y)
    {
        for(int x = 0; x < fc.container->width - fc.object->width;  x+=50)//++x)
        {
	  Pixel p(x,y);
	  uc diff = fc.getWindowDiff(p, fc.object->width, fc.object->height);
	  cout << (int(diff)) << ": \t" << p.toString() << endl;
	  /*
	    uc g = fc.getGrayScale(Pixel(x,y));
	    
	    int offset2 = (y * width + x) * 3;
	    grayData[offset2 + 0] = g;
	    grayData[offset2 + 1] = g;
	    grayData[offset2 + 2] = g;
	    */
        }
    }
    
    //stbi_write_bmp("test.bmp", width, height, 3, grayData);
}

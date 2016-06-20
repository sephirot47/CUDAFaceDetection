#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stbi.h"
#include "../include/stbi_write.h"

#define NUM_BLOCKS 255
#define NUM_THREADS 1024
#ifndef NUM_DEVICES
	#define NUM_DEVICES 4
#endif

#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
#define IMG_CHANNELS 3

//Optimal values 40, 550
#define THRESH_9x9 40     //Bigger = more restrictive
#define THRESH_30x30 550  //Bigger = less restrictive

typedef unsigned char uc;
using namespace std;

struct Pixel { int x,y; Pixel(int _x, int _y) {x=_x; y=_y;} };
struct Color { int r,g,b,a; Color(int _r, int _g, int _b, int _a) { r=_r; g=_g; b=_b; a=_a; } };
class Image
{
public:
    uc *data;
    char* filename;

    Image(char* _filename) {
        filename = _filename;
        FILE *file = fopen(_filename, "r");
        if (file != NULL) {
            data = stbi_load_from_file(file, &_width, &_height, &comp, IMG_CHANNELS); // rgb
            fclose (file);
            printf("%s read successfully\n", _filename);
        }
        else printf("Image '%s' not found\n", _filename);
    }

    int width() { return _width;  }
    int height() { return _height; }

    Color getColor(Pixel p) {
        int offset = (p.y * _width + p.x) * comp;
        return Color((data[offset + 0]), (data[offset + 1]),
                     (data[offset + 2]), (data[offset + 3]));
    }

    void setColor(Pixel p, Color c) {
        int offset = (p.y * _width + p.x) * 3;
        data[offset + 0] = c.r;
        data[offset + 1] = c.g;
        data[offset + 2] = c.b;
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
    Image *image;

    FaceDetection(char* imageFile)
    {
        image = new Image(imageFile);
    }

    void saveResult() {
        printf("Saving result...\n");

        Color c = Color(0,255,0,255);
        for(Box b : resultWindows) {
            for (int i = b.x; i < b.x+b.w; ++i) {
                image->setColor(Pixel(i,b.y),c);
                image->setColor(Pixel(i,b.y+1),c);
            }
            for (int i = b.x; i < b.x+b.w; ++i) {
                image->setColor(Pixel(i,b.y+b.h-1),c);
                image->setColor(Pixel(i,b.y+b.h-2),c);
            }
            for (int i = b.y; i < b.y+b.h; ++i) {
                image->setColor(Pixel(b.x,i),c);
                image->setColor(Pixel(b.x+1,i),c);
            }
            for (int i = b.y; i < b.y+b.h; ++i) {
                image->setColor(Pixel(b.x+b.w-1,i),c);
                image->setColor(Pixel(b.x+b.w-2,i),c);
            }
        }

        stbi_write_bmp("output/result.bmp", image->width(), image->height(), 3, image->data);
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


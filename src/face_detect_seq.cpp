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
#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
#define IMG_CHANNELS 3

//Optimal values 40, 550
#define THRESH_9x9 20     //Bigger = more restrictive
#define THRESH_30x30 475  //Bigger = less restrictive

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
                uc lowerLeft  = img[imgOffset + imgWidth - 1];
                uc lowerRight = img[imgOffset + imgWidth + 1];

                int sum = -upperLeft - upperRight - 2*up + 2*down + lowerLeft + lowerRight;

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

int getSecondStageHeuristic(uc *img) {
    int sumDiff    = 0;
    int leftEye    = getWindowMeanGS(img, 2, 4, 9, 5,30);
    int rightEye   = getWindowMeanGS(img,18, 4, 9, 5,30);
    int upperNose  = getWindowMeanGS(img,11, 1, 6,13,30);
    int lowerNose  = getWindowMeanGS(img,10,15, 9, 5,30);
    int leftCheek  = getWindowMeanGS(img, 1,10, 8,10,30);
    int rightCheek = getWindowMeanGS(img,19,10, 8,10,30);
    int mouth      = getWindowMeanGS(img, 8,21,13, 5,30);

    sumDiff += leftEye;
    sumDiff += rightEye;
    sumDiff += abs(leftEye - rightEye); // simmetry

    sumDiff += 255-upperNose;
    sumDiff += abs(125-lowerNose);

    sumDiff += 255-leftCheek;
    sumDiff += 255-rightCheek;
    sumDiff += abs(leftCheek - rightCheek); // simmetry

    sumDiff += mouth; // mouth

    return sumDiff;
}

int main(int argc, char** argv)
{
    FaceDetection fc(argv[1]);

    int numBytesContainer = fc.image->width() * fc.image->height() * sizeof(uc);

    printf("Container File: %s, size(%d px, %d px), bytes(%d B)\n",
	    fc.image->filename, fc.image->width(), fc.image->height(),
	    numBytesContainer);

    // Obtener Memoria en el host
    uc *h_imageImageGS = (uc*) malloc(numBytesContainer);
    printf("Filling ContainerGS in the host with GS values...\n");
    for(int y = 0; y < fc.image->height(); ++y) {
	for(int x = 0; x < fc.image->width(); ++x) {
	    h_imageImageGS[y * fc.image->width() + x] = fc.image->getGrayScale(Pixel(x,y));
	}
    }

    printf("Resizing original image....\n");
    int numBytesImage = IMG_WIDTH * IMG_WIDTH * sizeof(uc);
    uc *h_imageGS = (uc*) malloc(numBytesImage);
    resize(h_imageImageGS,
	    0, 0, fc.image->width(), fc.image->height(), fc.image->width(),
	    h_imageGS,
	    0, 0, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);

    //Original 100,100
    int winWidths[] = {35, 40, 45, 50, 55, 60, 65};
    //int winWidths[] = {35, 40, 45, 50, 55, 60, 65, 75, 85, 95, 105, 115, 125, 140, 150, 160, 170, 180, 190};
    int winHeights[] = {35, 40, 45, 50, 55, 60, 65};
    //int winHeights[] = {35, 40, 45, 50, 55, 60, 65, 75, 85, 95, 105, 115, 125, 140, 150, 160, 170, 180, 190};

    const int numWindowsWidth  = (sizeof(winWidths) / sizeof(int));
    const int numWindowsHeight = (sizeof(winHeights) / sizeof(int));

    float widthRatio = float(fc.image->width()) / IMG_WIDTH;
    float heightRatio = float(fc.image->height()) / IMG_HEIGHT;

    for(int blockId_x = 0; blockId_x < NUM_BLOCKS; ++blockId_x) {
    for(int blockId_y = 0; blockId_y < NUM_BLOCKS; ++blockId_y) {
    for(int i = 0; i < numWindowsWidth; ++i) {
    for(int j = 0; j < numWindowsHeight; ++j) {
	int winWidth = winWidths[i];
	int winHeight = winHeights[j];
	int xstep = (IMG_WIDTH - winWidth/widthRatio) / NUM_BLOCKS + 1;
	int ystep = (IMG_HEIGHT - winHeight/heightRatio) / NUM_BLOCKS + 1;
	int x = blockId_x * xstep;
	int y = blockId_y * ystep;
	
	/*if(winWidth == winWidths[0] && winHeight == winHeights[0] && x % 8 == 0 && y % 8 == 0)
	    printf("Computing x,y,w,h(%d,%d,%d,%d)\n", x, y, winWidth, winHeight);*/

	// FIRST HEURISTIC
	uc window9x9[9*9];
	resize(h_imageGS, x, y, winWidth, winHeight, IMG_WIDTH, window9x9, 0, 0, 9, 9, 9);
	histogramEqualization(window9x9, 0, 0, 9, 9, 9);

	uc hv1 = getFirstStageHeuristic(window9x9);
	if (hv1 >= THRESH_9x9)
	{
	    // SECOND HEURISTIC
	    uc *sobelImg = (uc*) malloc(winWidth * winHeight * sizeof(uc));
	    sobelEdgeDetection(h_imageGS, x, y, winWidth, winHeight, IMG_WIDTH, sobelImg);

	    uc window30x30[30*30];
	    resize(sobelImg, 0, 0, winWidth, winHeight, winWidth, window30x30, 0, 0, 30, 30, 30);
	    toBlackAndWhite(window30x30, 0, 0, 30, 30, 30);
	    free(sobelImg);
	    
	    int hv2 = getSecondStageHeuristic(window30x30);
	    if (hv2 <= THRESH_30x30)
	    {
		printf("Result found: %d, %d, %d, %d\n", x, y, winWidth, winHeight);
		fc.resultWindows.push_back( Box(int(x * widthRatio),
						int(y * heightRatio),
						int(winWidth * widthRatio),
						int(winHeight * heightRatio)));
	    }
	}
    }
    }
    }
    }

    fc.saveResult();
    printf("Result saved to 'output/result.bmp'\n");
}








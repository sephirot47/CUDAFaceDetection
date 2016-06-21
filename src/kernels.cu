#include "common.h"

__device__ __host__ uc getWindowMeanGS(uc *img, int ox, int oy, int winWidth, int winHeight, int imgWidth) {
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

// Resize always to a smaller size -> downsample
__device__ void resize(uc *src, int srcx, int srcy, int srcw, int srch, int srcTotalWidth, //x,y,width,height
                       uc *dst, int dstx, int dsty, int dstw, int dsth, int dstTotalWidth) //x,y,width,height
{
    float bw = float(srcw) / dstw;
    float bh = float(srch) / dsth;

    int size = dsth * dstw;
    for(int i = threadIdx.x; i < size; i += NUM_THREADS)
    {
        int dx = dstx + (i % dstw);
        int dy = dsty + (i / dstw);

        uc mean = getWindowMeanGS(src,
                                  srcx + ceil(dx*bw), srcy + ceil(dy*bh), //x, y
                                  floor(bw), floor(bh),                   //width height
                                  srcTotalWidth                           //totalWidth
                                  );

        dst[dy * dstTotalWidth + dx] = mean;
    }
}


__device__ void getHistogram(uc *img, int ox, int oy, int width, int height, int imgWidth, float histogram[256]) {

    float npixels = width * height;
    float unitProb = 1.0f/npixels;

    if(threadIdx.x < 256) histogram[threadIdx.x] = 0;

    for(int i = threadIdx.x; i < npixels; i += NUM_THREADS)
    {
        int wx = i % width;
        int wy = i / width;
        int offset = (oy + wy) * imgWidth + (ox + wx);
        uc v = img[offset];
        atomicAdd(&histogram[v], unitProb);
    }
}

// Increase contrast
__device__ void histogramEqualization(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
    __shared__ float histogram[256];
    __shared__ float accumulatedProbs[256];
    
    getHistogram(img, ox, oy, width, height, imgWidth, histogram);
    
    if(threadIdx.x == 0)
    { 
        accumulatedProbs[0] = histogram[0];
    	for(int i = 1; i < 256; ++i)
            accumulatedProbs[i] = accumulatedProbs[i-1] + histogram[i];
    }
    __syncthreads();
    
    
    int size = width * height;
    for(int i = threadIdx.x; i < size; i += NUM_THREADS)
    {
	 int wx = i % width;
	 int wy = i / width;
	 int offset = (oy + wy) * imgWidth + (ox + wx);
    	 uc v = img[offset];
         img[offset] = floor(255 * accumulatedProbs[v]);
    }
}

__device__ void toBlackAndWhite(uc *img, int ox, int oy, int width, int height, int imgWidth)
{
    int size = width * height;
    for(int i = threadIdx.x; i < size; i += NUM_THREADS)
    {
        int wx = i % width;
        int wy = i / width;
        int offset = (oy + wy) * width + (ox + wx);
        uc v = img[offset];
        img[offset] = v > 200 ? 255 : 0;
    }
}

__device__ uc getFirstStageHeuristic(uc *img) {
    int v = img[22] - (img[19]+img[20]+img[24]+img[25]+img[58])/5;
    return v < 0 ? 0 : v;
}

// Find edges in horizontal direction
__device__ void sobelEdgeDetection(uc *img, int ox, int oy,  int winWidth, int winHeight, int imgWidth, uc *sobelImg)
{
    uc threshold = 24;

    int size = winWidth * winHeight;
    for (int i = threadIdx.x; i < size; i+=NUM_THREADS)
    {
        int wx = i % winWidth;
        int wy = i / winWidth;
        int winOffset = wy * winWidth + wx;

        int x = ox + wx;
        int y = oy + wy;
        int imgOffset = y * imgWidth + x;

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

__device__ int getSecondStageHeuristic(uc *img) {
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

__global__ void detectFaces(uc *img, int winWidth, int winHeight, uc  *resultMatrix, int resultIndex)
{
    int xstep = (IMG_WIDTH - winWidth) / NUM_BLOCKS + 1;
    int ystep = (IMG_HEIGHT - winHeight) / NUM_BLOCKS + 1;

    // Window origin
    int x = blockIdx.x * xstep;
    int y = blockIdx.y * ystep;
    int blockId = blockIdx.y * NUM_BLOCKS + blockIdx.x;
    int resultOffset = resultIndex * NUM_BLOCKS * NUM_BLOCKS;

    if(x + winWidth > IMG_WIDTH || y + winHeight > IMG_HEIGHT)
    {
        resultMatrix[blockId + resultOffset] = 0;
        return;
    }

    // FIRST HEURISTIC
    __shared__ uc window30x30[30*30];
    resize(img,
           x, y, winWidth, winHeight, IMG_WIDTH,
           window30x30,
           0, 0, 9, 9, 9);
    __syncthreads();

    histogramEqualization(window30x30, 0, 0, 9, 9, 9);
    __syncthreads();
    
    __shared__ uc hv1;
    if(threadIdx.x == 0) {
        hv1 = getFirstStageHeuristic(window30x30);
    }
    __syncthreads();

    if (hv1 >= THRESH_9x9)
    {
        // SECOND HEURISTIC
        __shared__ uc sobelImg[200*200];
        sobelEdgeDetection(img, x, y, winWidth, winHeight, IMG_WIDTH, sobelImg);
        __syncthreads();

	resize(sobelImg,
               0, 0, winWidth, winHeight, winWidth,
               window30x30,
               0, 0, 30, 30, 30);
        __syncthreads();

        toBlackAndWhite(window30x30, 0, 0, 30, 30, 30);
        __syncthreads();

        if(threadIdx.x == 0) {
            int hv2 = getSecondStageHeuristic(window30x30);
            if (hv2 <= THRESH_30x30)
            {
                // Save result! We detected a face yayy
                resultMatrix[blockId + resultOffset] = 1;
            }
            else resultMatrix[blockId + resultOffset] = 0;
        }
    }
    else resultMatrix[blockId + resultOffset] = 0;
}


__global__ void detectFaces(uc *img, int winWidth, int winHeight, uc  *resultMatrix)
{
    int xstep = (IMG_WIDTH - winWidth) / NUM_BLOCKS + 1;
    int ystep = (IMG_HEIGHT - winHeight) / NUM_BLOCKS + 1;

    // Window origin
    int x = blockIdx.x * xstep;
    int y = blockIdx.y * ystep;
    int blockId = blockIdx.y * NUM_BLOCKS + blockIdx.x;

    if(x + winWidth > IMG_WIDTH || y + winHeight > IMG_HEIGHT)
    {
        resultMatrix[blockId] = 0;
        return;
    }

    // FIRST HEURISTIC
    __shared__ uc window30x30[30*30];
    resize(img,
           x, y, winWidth, winHeight, IMG_WIDTH,
           window30x30,
           0, 0, 9, 9, 9);
    __syncthreads();

    histogramEqualization(window30x30, 0, 0, 9, 9, 9);
    __syncthreads();
    
    __shared__ uc hv1;
    if(threadIdx.x == 0) {
        hv1 = getFirstStageHeuristic(window30x30);
    }
    __syncthreads();

    if (hv1 >= THRESH_9x9)
    {
        // SECOND HEURISTIC
        __shared__ uc sobelImg[200*200];
        sobelEdgeDetection(img, x, y, winWidth, winHeight, IMG_WIDTH, sobelImg);
        __syncthreads();

	resize(sobelImg,
               0, 0, winWidth, winHeight, winWidth,
               window30x30,
               0, 0, 30, 30, 30);
        __syncthreads();

        toBlackAndWhite(window30x30, 0, 0, 30, 30, 30);
        __syncthreads();

        if(threadIdx.x == 0) {
            int hv2 = getSecondStageHeuristic(window30x30);
            if (hv2 <= THRESH_30x30)
            {
                // Save result! We detected a face yayy
                resultMatrix[blockId] = 1;
            }
            else resultMatrix[blockId] = 0;
        }
    }
    else resultMatrix[blockId] = 0;
}


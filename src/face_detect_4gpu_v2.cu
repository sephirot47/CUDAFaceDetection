#include "common.h"
#include "kernels.cu"

void resize_seq(uc *src, int srcx, int srcy, int srcw, int srch, int srcTotalWidth, //x,y,width,height
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
            //Save in its position the mean of the corresponding window pixels
            uc mean = getWindowMeanGS(src,
                                      srcx + ceil(dx*bw), srcy + ceil(dy*bh), //x, y
                                      floor(bw), floor(bh),                   //width height
                                      srcTotalWidth                           //totalWidth
                                      );

            dst[dy * dstTotalWidth + dx] = mean;
        }
    }
}

__global__ void detectFaces(uc *img, int winWidth, int winHeight, uc  *resultMatrix, int resultIndex);

void CheckCudaError(int line) {
    cudaError_t error;
    error = cudaGetLastError();
    if (error) {
        printf("(ERROR) - %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, line);
        exit(EXIT_FAILURE);
    }
}

#define CE() { CheckCudaError(__LINE__); }

int main(int argc, char** argv)
{
    cout << "Usage: " << argv[0] << " <image file name>" << endl;
    for (int i = 0; i < argc; ++i) { cout << argv[i] << endl; }

    // Read input image
    FaceDetection fc(argv[1]);
    printf("image File: %s, size(%d px, %d px)\n",
           fc.image->filename, fc.image->width(), fc.image->height());

    // Convert input image to grayscale
    int numBytesImageOriginal = fc.image->width() * fc.image->height() * sizeof(uc);
    uc *h_imageGSOriginal = (uc*) malloc(numBytesImageOriginal);
    printf("Adapting input. Creating grayscale image....\n");
    for(int y = 0; y < fc.image->height(); ++y) {
        for(int x = 0; x < fc.image->width(); ++x) {
            h_imageGSOriginal[y * fc.image->width() + x] = fc.image->getGrayScale(Pixel(x,y));
        }
    }

    // Resize input image
    printf("Resizing original image....\n");
    int numBytesImage = IMG_WIDTH * IMG_HEIGHT * sizeof(uc);
    uc *h_imageGS = (uc*) malloc(numBytesImage);
    resize_seq(h_imageGSOriginal,
               0, 0, fc.image->width(), fc.image->height(), fc.image->width(),
               h_imageGS,
               0, 0, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);


    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < NUM_DEVICES) { printf("Not enough GPUs\n"); exit(-1); }

    dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);
    int winWidths[] = {35, 40, 45, 50, 55, 60, 65, 70, 75, 85, 95, 105, 115, 125, 140, 150, 160, 170, 180, 190};
    int winHeights[] = {35, 40, 45, 50, 55, 60, 65, 70, 75, 85, 95, 105, 115, 125, 140, 150, 160, 170, 180, 190};

    const int numWindowWidths  = (sizeof(winWidths) / sizeof(int));
    const int numWindowHeights = (sizeof(winHeights) / sizeof(int));
    const int numWindows = numWindowWidths * numWindowHeights;
    const int windowsPerDevice = numWindows / NUM_DEVICES;
    const float widthRatio =  float(fc.image->width())/IMG_WIDTH;
    const float heightRatio =  float(fc.image->height())/IMG_HEIGHT;

    printf("Num windows: %i\n", numWindows);
    printf("Windows per device: %i\n", windowsPerDevice);


    // Get host memory
    printf("Getting memory in the host to allocate resultMatrix...\n");
    int numBytesResultMatrix = NUM_BLOCKS * NUM_BLOCKS * windowsPerDevice * sizeof(uc);
    uc *h_resultMatrix[NUM_DEVICES];
    for(int i = 0; i < NUM_DEVICES; ++i)
        h_resultMatrix[i] = (uc*) malloc(numBytesResultMatrix);

    // Get memory in every device
    uc *d_imageGS[NUM_DEVICES], *d_resultMatrix[NUM_DEVICES];
    for(int i = 0; i < NUM_DEVICES; ++i)
    {
        printf("Getting memory in device %d...\n", i);
        cudaSetDevice(i);
        #ifndef USE_PINNED
        cudaMalloc((uc**)&d_imageGS[i], numBytesImage); CE();
        cudaMalloc((uc**)&d_resultMatrix[i], numBytesResultMatrix); CE();
        #else
        cudaMallocHost((uc**)&d_imageGS[i], numBytesImage); CE();
        cudaMallocHost((uc**)&d_resultMatrix[i], numBytesResultMatrix); CE();
        #endif
    }


    // Copy data from host to device, execute kernel
    cudaSetDevice(0);
    cudaEvent_t E0, E1;
    cudaEventCreate(&E0);
    cudaEventCreate(&E1);

    cudaEventRecord(E0, 0);

    for(int i = 0; i < NUM_DEVICES; ++i)
    {
        cudaSetDevice(i);
        cudaMemcpyAsync(d_imageGS[i], h_imageGS, numBytesImage, cudaMemcpyHostToDevice); CE();
        for(int j = 0; j < windowsPerDevice; ++j)
        {
            int index = (i*windowsPerDevice + j);
            int wi = index / numWindowWidths;
            int hi = index % numWindowHeights;
            detectFaces<<<dimGrid, dimBlock>>>(d_imageGS[i],
                                               winWidths[wi] / widthRatio,
                                               winHeights[hi] / heightRatio,
                                               d_resultMatrix[i],
                                               j); CE();
        }
    }

    // Retrieve results from device to host
    for(int i = 0; i < NUM_DEVICES; ++i)
    {
        cudaSetDevice(i);
        cudaMemcpyAsync(h_resultMatrix[i], d_resultMatrix[i], numBytesResultMatrix, cudaMemcpyDeviceToHost);
    }

    for(int i = 0; i < NUM_DEVICES; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    cudaSetDevice(0);
    cudaEventRecord(E1, 0);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,  E0, E1);
    printf("Kernel elapsed time: %4.6fs\n", elapsedTime / 1000.0);


    // Process results
    for(int k = 0; k < numWindows; ++k)
    {
    for(int i = 0; i < NUM_BLOCKS; ++i)
    {
    for(int j = 0; j < NUM_BLOCKS; ++j)
    {
        int device = k / windowsPerDevice;
        int resultIndex = k % windowsPerDevice;
        int resultOffset = resultIndex * NUM_BLOCKS * NUM_BLOCKS;
        if (h_resultMatrix[device][i * NUM_BLOCKS + j + resultOffset] == 1)
        {
            int wi = k / numWindowWidths;
            int hi = k % numWindowHeights;
            int kernelStepWidth = (IMG_WIDTH - winWidths[wi]/widthRatio) / NUM_BLOCKS + 1;
            int kernelStepHeight = (IMG_HEIGHT - winHeights[hi]/heightRatio) / NUM_BLOCKS + 1;
            printf("Result found for size(%d,%d) in x,y: (%d,%d)\n", winWidths[wi], winHeights[hi], j, i);
            fc.resultWindows.push_back(Box(int(j * kernelStepWidth * widthRatio),
                                           int(i * kernelStepHeight * heightRatio),
                                           int(winWidths[wi]),
                                           int(winHeights[hi])));
        }
    }
    }
    }
    fc.saveResult();


    // Free memory in every device
    printf("Freeing device memory...\n");
    for(int i = 0; i < NUM_DEVICES; ++i)
    {
    cudaSetDevice(i);
    cudaFree(d_imageGS[i]);
    cudaFree(d_resultMatrix[i]);
    }

    printf("Done.\n");
}

// Authored by TomHeaven, hanlin_tan@nudt.edu.cn, 2016.12.02

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cmath>

//#define DEBUG

/**
Device code
*/

// check error
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
  if (code != cudaSuccess)
  {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  } else {
    #ifdef DEBUG
    printf("cuda returned code == cudaSuccess\n");
    #endif
  }
}


/**
% input:
% startThreadNum, one task may require several launches, this param records the start thread num to continue woring.
% image, vector form of image with size of x_height * x_width
% x_height, x_width, size of image
% output:
% sumImg, output vector with size of x_height * x_width
% d_info, vector for debug
*/
void __global__  winSumFilterKernel(const int startThreadNum, const double* image, const int x_height, const int x_width, const int radius, double* sumImg, float* d_info) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x + startThreadNum;
  int i = idx / x_width;
  int j = idx % x_width;

  int off = idx; // i + j * x_height; //zero based
  int rmin = max(i - radius, 0);
  int rmax = min(i + radius, x_height-1);
  int cmin = max(j - radius, 0);
  int cmax = min(j + radius, x_width -1);

  #ifdef DEBUG
  if (idx == 0) {
    d_info[0] = idx;
    d_info[1] = i;
    d_info[2] = j;
    d_info[3] = off;
    d_info[4] = rmin;
    d_info[5] = rmax;
    d_info[6] = cmin;
    d_info[7] = cmax;
    d_info[8] = x_height;
    d_info[9] = x_width;
    d_info[10] = radius;
  }
  #endif
    double s = 0;
  for(int y = cmin; y <= cmax; y++) {
    for(int x = rmin; x <= rmax; x++)
    {
      int off_tmp = x*x_width  + y;
      s += image[off_tmp];
    }
  }
  sumImg[off] = s;
}

/*
* Host code
*/
void mexFunction(int nlhs, mxArray *plhs[],
  int nrhs, mxArray const *prhs[])
  {
    /* Declare all variables.*/
    mxGPUArray const * image;
    mxGPUArray * sumImg;

    double const * d_image;
    double * d_sumImg;
    double * ptr;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Choose a reasonably sized number of threads for the block. */
    // Don't set threadsPerBlock too big or the shared memory may exceed block
    // shared memory limit and cause CUDA_ILLEGAL_ADDRESS error !!!
    int const threadsPerBlock = 32;
    int blocksPerGrid = 1024;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs != 3) || !(mxIsGPUArray(prhs[0]))) {
      mexErrMsgIdAndTxt(errId, errMsg);
    }

    image = mxGPUCreateFromMxArray(prhs[0]);

    ptr = mxGetPr(prhs[1]);
    int x_height = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[2]);
    int radius = int(ptr[0] + 0.5);


    int X_length = (int)(mxGPUGetNumberOfElements(image));
    int x_width = X_length / x_height;

    #ifdef DEBUG
    printf("nrhs = %d, x_height = %d, radius = %d\n", nrhs, x_height, radius);
    #endif

    #ifdef DEBUG
    printf("mxGPUGetClassID(A) = %d, mxDOUBLE_CLASS = %d\n", mxGPUGetClassID(image), mxDOUBLE_CLASS);
    #endif

    // Verify that X really is a double array before extracting the pointer.
    if (mxGPUGetClassID(image) != mxDOUBLE_CLASS) {
      mexErrMsgIdAndTxt(errId, errMsg);
    }

    #ifdef DEBUG
    printf("break point 1\n");
    #endif

    /*  Extract a pointer to the input data on the device. */
    d_image = (double const *)(mxGPUGetDataReadOnly(image));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    sumImg = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(image),
    mxGPUGetDimensions(image),
    mxGPUGetClassID(image),
    mxGPUGetComplexity(image),
    MX_GPU_DO_NOT_INITIALIZE);
    d_sumImg = (double *)(mxGPUGetData(sumImg));

    #ifdef DEBUG
    printf("break point 2\n");
    #endif

    float* d_info = NULL;
    float info[100];
    cudaMalloc((void **)&d_info, 100 *sizeof(float));
    cudaMemcpy(d_info, info, 100 *sizeof(float), cudaMemcpyHostToDevice);
    #ifdef DEBUG
    printf("bpg = %d, tpb = %d\n", blocksPerGrid, threadsPerBlock);
    #endif

    int thread_num = blocksPerGrid * threadsPerBlock;
    for(int startThreadNum = 0; startThreadNum < x_width*x_height; startThreadNum += thread_num) {
      // minKernel(const int startThreadNum, const double* r, onst double* g，onst double* b, const int x_height, const int x_width, const int wnd, double* darkChannel, float* d_info)
      winSumFilterKernel<<<blocksPerGrid, threadsPerBlock>>>(startThreadNum, d_image, x_height, x_width, radius, d_sumImg, d_info);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    }
    #ifdef DEBUG
    printf("break point 3\n");
    cudaMemcpy(info, d_info, 100 *sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 11; ++i)
    printf("info[%d] = %f\n", i, info[i]);
    #endif
    cudaFree(d_info);
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(sumImg);
    /*
    * The mxGPUArray pointers are host-side structures that refer to device
    * data. These must be destroyed before leaving the MEX function.
    */
    mxGPUDestroyGPUArray(image);
    mxGPUDestroyGPUArray(sumImg);
  }

// Authored by TomHeaven, hanlin_tan@nudt.edu.cn, 2016.12.02

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cmath>

//#define DEBUG

/**
Device code
*/

// Set this const according to b2
#define B2 7
// Set this const according to wnd
#define WND 10
// derived const for vector size in kernel function
#define VEC_SIZE ((WND+WND+1)*(WND+WND+1))

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
% r,g,b, vector form of R,G,B image with size of x_height * x_width
% x_height, x_width, size of R,G,B image
% output:
% darkChannel, output vector with size of x_height * x_width
% d_info, vector for debug
*/
void __global__  minKernel(const int startThreadNum, const double* r, const double* g, const double* b, const int x_height, const int x_width, const int wnd, double* darkChannel, float* d_info) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x + startThreadNum;
  int i = idx / x_width;
  int j = idx % x_width;

  int off = idx; // i + j * x_height; //zero based
  int rmin = max(i - wnd/2, 0);
  int rmax = min(i + wnd/2, x_height-1);
  int cmin = max(j - wnd/2, 0);
  int cmax = min(j + wnd/2, x_width -1);

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
    d_info[10] = wnd;
  }
  #endif
  double minValue = 99999;
  for(int y = cmin; y <= cmax; y++) {
    for(int x = rmin; x <= rmax; x++)
    {
      //int off_tmp = x  + y * x_height;
      int off_tmp = x*x_width  + y;
      minValue = r[off_tmp] < minValue ? r[off_tmp] : minValue;
      minValue = g[off_tmp] < minValue ? g[off_tmp] : minValue;
      minValue = b[off_tmp] < minValue ? b[off_tmp] : minValue;
    }
  }
  darkChannel[off] = minValue;
}

/*
* Host code
*/
void mexFunction(int nlhs, mxArray *plhs[],
  int nrhs, mxArray const *prhs[])
  {
    /* Declare all variables.*/
    mxGPUArray const *r, *g, *b;
    mxGPUArray * darkChannel;

    double const * d_r, *d_g, *d_b;
    double * d_darkChannel;
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
    if ((nrhs != 5) || !(mxIsGPUArray(prhs[0]))) {
      mexErrMsgIdAndTxt(errId, errMsg);
    }

    r = mxGPUCreateFromMxArray(prhs[0]);
    g = mxGPUCreateFromMxArray(prhs[1]);
    b = mxGPUCreateFromMxArray(prhs[2]);

    ptr = mxGetPr(prhs[3]);
    int x_height = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[4]);
    int win_size = int(ptr[0] + 0.5);


    int X_length = (int)(mxGPUGetNumberOfElements(r));
    int x_width = X_length / x_height;

    #ifdef DEBUG
    printf("nrhs = %d, x_height = %d, win_size = %d\n", nrhs, x_height, win_size);
    #endif

    #ifdef DEBUG
    printf("mxGPUGetClassID(A) = %d, mxDOUBLE_CLASS = %d\n", mxGPUGetClassID(r), mxDOUBLE_CLASS);
    #endif

    // Verify that X really is a double array before extracting the pointer.
    if (mxGPUGetClassID(r) != mxDOUBLE_CLASS) {
      mexErrMsgIdAndTxt(errId, errMsg);
    }

    #ifdef DEBUG
    printf("break point 1\n");
    #endif

    /*  Extract a pointer to the input data on the device. */
    d_r = (double const *)(mxGPUGetDataReadOnly(r));
    d_g = (double const *)(mxGPUGetDataReadOnly(g));
    d_b = (double const *)(mxGPUGetDataReadOnly(b));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    darkChannel = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(r),
    mxGPUGetDimensions(r),
    mxGPUGetClassID(r),
    mxGPUGetComplexity(r),
    MX_GPU_DO_NOT_INITIALIZE);
    d_darkChannel = (double *)(mxGPUGetData(darkChannel));

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
      minKernel<<<blocksPerGrid, threadsPerBlock>>>(startThreadNum, d_r, d_g, d_b, x_height, x_width,  win_size, d_darkChannel, d_info);
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
    plhs[0] = mxGPUCreateMxArrayOnGPU(darkChannel);
    /*
    * The mxGPUArray pointers are host-side structures that refer to device
    * data. These must be destroyed before leaving the MEX function.
    */
    mxGPUDestroyGPUArray(r);
    mxGPUDestroyGPUArray(g);
    mxGPUDestroyGPUArray(b);
    mxGPUDestroyGPUArray(darkChannel);
  }

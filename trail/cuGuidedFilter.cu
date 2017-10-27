// Authored by TomHeaven, hanlin_tan@nudt.edu.cn, 2016.12.02

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cmath>
#include <cstdio>

//#define DEBUG

int const threadsPerBlock = 32;
int blocksPerGrid = 1024;
int X_LENGTH;
int thread_num;


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

// Kernel definition
__global__ void vecAddKernel(const int startThreadNum, double* A, double* B, double* C)
{
    int i = threadIdx.x + startThreadNum;
    C[i] = A[i] + B[i];
}

// Kernel definition
__global__ void vecDifKernel(const int startThreadNum, double* A, double* B, double* C)
{
    int i = threadIdx.x + startThreadNum;
    C[i] = A[i] - B[i];
}

// Kernel definition
__global__ void vecMulKernel(const int startThreadNum, const double* A, const double* B, double* C)
{
    int i = threadIdx.x + startThreadNum;
    C[i] = A[i] * B[i];
}

// Kernel definition
__global__ void vecDivKernel(const int startThreadNum, double* A, double* B, double* C)
{
    int i = threadIdx.x + startThreadNum;
    C[i] = A[i] / B[i];
}

void vecAdd(double*A, double *B, double* C) {
    for(int startThreadNum = 0; startThreadNum < X_LENGTH; startThreadNum += thread_num) {
      vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(startThreadNum, A, B, C);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    }
}

void vecDif(double*A, double *B, double* C) {
    for(int startThreadNum = 0; startThreadNum < X_LENGTH; startThreadNum += thread_num) {
      vecDifKernel<<<blocksPerGrid, threadsPerBlock>>>(startThreadNum, A, B, C);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    }
}

void vecMul(const double*A, const double *B, double* C) {
    for(int startThreadNum = 0; startThreadNum < X_LENGTH; startThreadNum += thread_num) {
      vecMulKernel<<<blocksPerGrid, threadsPerBlock>>>(startThreadNum, A, B, C);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    }
}

void vecDiv(double*A, double *B, double* C) {
    for(int startThreadNum = 0; startThreadNum < X_LENGTH; startThreadNum += thread_num) {
      vecDivKernel<<<blocksPerGrid, threadsPerBlock>>>(startThreadNum, A, B, C);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    }
}

void winSumFilter(const double* d_image, int x_height, int x_width, int radius, double* d_avg, float* d_info) {
  for(int startThreadNum = 0; startThreadNum < x_width*x_height; startThreadNum += thread_num) {
    // minKernel(const int startThreadNum, const double* r, onst double* g，onst double* b, const int x_height, const int x_width, const int wnd, double* darkChannel, float* d_info)
    winSumFilterKernel<<<blocksPerGrid, threadsPerBlock>>>(startThreadNum, d_image, x_height, x_width, radius, d_avg, d_info);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
}


/*
* Host code
*/
void mexFunction(int nlhs, mxArray *plhs[],
  int nrhs, mxArray const *prhs[])
  {
    /* Declare all variables.*/
    mxGPUArray const * guide, *target;
    mxGPUArray * res;

    double const * d_guide , *d_target;
    double * d_res;
    double * ptr;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Choose a reasonably sized number of threads for the block. */
    // Don't set threadsPerBlock too big or the shared memory may exceed block
    // shared memory limit and cause CUDA_ILLEGAL_ADDRESS error !!!

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs != 5) || !(mxIsGPUArray(prhs[0]))) {
      mexErrMsgIdAndTxt(errId, errMsg);
    }

    guide = mxGPUCreateFromMxArray(prhs[0]);
    target = mxGPUCreateFromMxArray(prhs[1]);

    ptr = mxGetPr(prhs[2]);
    int radius = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[3]);
    double eps = double(ptr[0]);
    ptr = mxGetPr(prhs[4]);
    int x_height = int(ptr[0] + 0.5);

    int X_length = (int)(mxGPUGetNumberOfElements(guide));
    X_LENGTH = X_length;
    int x_width = X_length / x_height;

    #ifdef DEBUG
    printf("nrhs = %d, x_height = %d, radius = %d\n", nrhs, x_height, radius);
    #endif

    #ifdef DEBUG
    printf("mxGPUGetClassID(A) = %d, mxDOUBLE_CLASS = %d\n", mxGPUGetClassID(guide), mxDOUBLE_CLASS);
    #endif

    // Verify that X really is a double array before extracting the pointer.
    if (mxGPUGetClassID(guide) != mxDOUBLE_CLASS) {
      mexErrMsgIdAndTxt(errId, errMsg);
    }

    #ifdef DEBUG
    printf("break point 1\n");
    #endif

    /*  Extract a pointer to the input data on the device. */
    d_guide = (double const *)(mxGPUGetDataReadOnly(guide));
    d_target = (double const *)(mxGPUGetDataReadOnly(target));
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    res = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(guide),
    mxGPUGetDimensions(guide),
    mxGPUGetClassID(guide),
    mxGPUGetComplexity(guide),
    MX_GPU_DO_NOT_INITIALIZE);

    d_res = (double *)(mxGPUGetData(res));

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

    thread_num = blocksPerGrid * threadsPerBlock;

    double* ones, *d_ones, *d_mean_g, *d_mean_t, *d_corr_gg, *d_corr_gt, *d_tmp, *d_tmp2, *d_avg;

    const int MEMSIZE = x_width * x_height* sizeof(double);
    ones = (double*)malloc(MEMSIZE);
    cudaMalloc((void **)&d_ones, MEMSIZE);
    for(int i = 0; i < x_width * x_height; ++i)
       ones[i] = 1;
    cudaMemcpy(d_ones, ones, MEMSIZE, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_mean_g, MEMSIZE);
    cudaMalloc((void**)&d_mean_t, MEMSIZE);
    cudaMalloc((void**)&d_corr_gg, MEMSIZE);
    cudaMalloc((void**)&d_corr_gt, MEMSIZE);
    cudaMalloc((void**)&d_tmp, MEMSIZE);
    cudaMalloc((void**)&d_tmp2, MEMSIZE);
    cudaMalloc((void**)&d_avg, MEMSIZE);
    // avg_denom
    winSumFilter( d_ones, x_height, x_width, radius, d_avg, d_info);
    // mean_g & mean_t
    winSumFilter( d_guide, x_height, x_width, radius, d_mean_g, d_info);
    vecDiv(d_mean_g, d_avg, d_mean_g);
    winSumFilter( d_target, x_height, x_width, radius, d_mean_t, d_info);
    vecDiv(d_mean_t, d_avg, d_mean_t);
    // corr_gg & corr_gt
    vecMul( d_guide, d_guide, d_tmp);
    winSumFilter( d_tmp, x_height, x_width, radius, d_corr_gg, d_info);
    vecMul( d_guide, d_target, d_tmp);
    winSumFilter( d_tmp, x_height, x_width, radius, d_corr_gt, d_info);
    // var_g & cov_gt --> tmp, tmp2
    vecMul( d_mean_g, d_mean_g, d_tmp);
    vecDif( d_corr_gg, d_tmp, d_tmp);
    vecMul( d_mean_g, d_mean_t, d_tmp2);
    vecDif( d_corr_gt, d_tmp2, d_tmp2);
    // a & b --> tmp, tmp2
    for(int i = 0; i < x_height * x_width; ++i)
       ones[i] = eps;
    cudaMemcpy(d_ones, ones, MEMSIZE, cudaMemcpyHostToDevice);
    vecAdd( d_tmp, d_ones, d_tmp);
    vecDiv( d_tmp2, d_tmp, d_tmp);
    // b --> tmp2
    vecMul( d_tmp, d_mean_g, d_tmp2);
    vecDif( d_mean_t, d_tmp2, d_tmp2);
    // mean_a mean_b --> mean_g, mean_t
    winSumFilter( d_tmp, x_height, x_width, radius, d_mean_g, d_info);
    vecDiv(d_mean_g, d_avg, d_mean_g);
    // mean_b --> mean_t
    winSumFilter( d_tmp, x_height, x_width, radius, d_mean_t, d_info);
    vecDiv(d_mean_t, d_avg, d_mean_t);
    // result
    vecMul(d_mean_t, d_guide, d_mean_t);
    vecAdd(d_mean_t, d_mean_g, d_res);

    #ifdef DEBUG
    printf("break point 3\n");
    cudaMemcpy(info, d_info, 100 *sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 11; ++i)
    printf("info[%d] = %f\n", i, info[i]);
    #endif
    cudaFree(d_info);
    free(ones);
    cudaFree(d_ones);
    cudaFree(d_mean_g);
    cudaFree(d_mean_t);
    cudaFree(d_corr_gg);
    cudaFree(d_corr_gt);
    cudaFree(d_tmp);
    cudaFree(d_tmp2);
    cudaFree(d_avg);
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(res);
    /*
    * The mxGPUArray pointers are host-side structures that refer to device
    * data. These must be destroyed before leaving the MEX function.
    */
    mxGPUDestroyGPUArray(guide);
    mxGPUDestroyGPUArray(target);
    mxGPUDestroyGPUArray(res);
  }

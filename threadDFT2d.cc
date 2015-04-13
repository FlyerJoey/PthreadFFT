// Threaded two-dimensional Discrete FFT transform
// Jian Yuan
// ECE8893 Project 2

#include <iostream>
#include <string>
#include <math.h>

#include "Complex.h"
#include "InputImage.h"
#include <pthread.h>

// You will likely need global variables indicating how
// many threads there are, and a Complex* that points to the
// 2d image being transformed.

using namespace std;

Complex* WNk = NULL;
Complex* imageData = NULL;
int imageWidth;
int imageHeight;

// Number of threads
const int nThreads = 16;

// For Barrier
int Count;                     // Number of threads presently in the barrier
pthread_mutex_t countMutex;
bool* localSense;              // Create an array of bool, one per thread
bool globalSense;              // Global sense

// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.
unsigned ReverseBits(unsigned v) { //  Provided to students
  unsigned n = imageWidth; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value
   
  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;        // Shift return value
      r |= (v & 0x1); // Merge in next bit
      v >>= 1;        // Shift reversal value
    }
  return r;
}

void ReverseArray(Complex* data, int N) {
	int index_r;
	for (int index = 0; index < N; index++) {
		index_r = ReverseBits(index);
		if (index < index_r)
			swap(data[index], data[index_r]);
	}
}

void pComputeW(int N) {
	// Precomputing the Wn values
	WNk = new Complex[N / 2];
	for (int k = 0; k < N / 2; k++) {
		WNk[k] = Complex(cos(2 * M_PI * k / N), (-1) * sin(2 * M_PI * k / N));
	}
}

// GRAD Students implement the following 2 functions.
// Undergrads can use the built-in barriers in pthreads.

// Call MyBarrier_Init once in main
void MyBarrier_Init(int N) { // you will likely need some parameters)
	Count = N;
	// Initialize the mutex used for FetchAndIncrement
	pthread_mutex_init(&countMutex, 0);
	// Create and initialize the localSense arrar, 1 entry per thread
	localSense = new bool[N];
	for (int i = 0; i < N; ++i) localSense[i] = true;
	// Initialize global sense
	globalSense = true;
}

int FetchAndDecrementCount() {
	// We don't have an atomic FetchAndDecrement, but we can get t the
	// same behavior by using a mutex
	pthread_mutex_lock(&countMutex);
	int myCount = Count;
	Count--;
	pthread_mutex_unlock(&countMutex);
	return myCount;
}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier(int myId, int N) { // Again likely need parameters 
	localSense[myId] = !localSense[myId];     // Toggle private sense variable
	if (FetchAndDecrementCount() == 1) {
		Count = N;
		globalSense = localSense[myId];
	}
	else {
		while (globalSense != localSense[myId]) {};
	}
}

void Transpose(Complex* data, int nColumns, int nRows) {
	Complex* aux = new Complex[nColumns * nRows];
	for (int i = 0; i < nColumns; i++) {
		for (int j = 0; j < nRows; j++) {
			aux[j + i * nRows] = data[i + j * nColumns];
		}
	}
	for (int i = 0; i < nColumns * nRows; i++) {
		data[i] = aux[i];
	}
	delete [] aux;
}
                    
void Transform1D(Complex* h, int N) {
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)
	for (int Np = 2; Np <= N; Np *= 2) {
		for (int w = 0; w < N / Np; w++) {
			for (int k = 0; k < Np / 2; k++) {
				Complex tmp = h[w * Np + k];
				h[w * Np + k] = h[w * Np + k] + WNk[k * N / Np] * h[w * Np + k + Np / 2];
				h[w * Np + k + Np / 2] = tmp - WNk[k * N / Np] * h[w * Np + k + Np / 2];
			}
		}
	}
}

void iTransform1D(Complex* h, int N) {
	for (int Np = N; Np >= 2; Np /= 2) {
		for (int w = 0; w < N / Np; w++) {
			for (int k = 0; k < Np / 2; k++) {
				Complex tmp = h[w * Np + k];
				h[w * Np + k] = (h[w * Np + k] + h[w * Np + k + Np / 2]) * 0.5;
				h[w * Np + k + Np / 2] = (tmp - h[w * Np + k + Np / 2]) * WNk[k * N / Np].Conj() * 0.5;
			}
		}
	}
}

// Child thread to do transform
void* Transform2DThread(void* v) {
	// This is the thread startign point.  "v" is the thread number
	// Calculate 1d DFT for assigned rows
	// wait for all to complete
	// Calculate 1d DFT for assigned columns
	// Decrement active count and signal main if all complete
	long myId = (long)v;

	int rowsPerThread = imageHeight / nThreads;
	int startingRow = myId * rowsPerThread;

	for (int r = 0; r < rowsPerThread; ++r) {
		int thisRow = startingRow + r;
		ReverseArray(&imageData[thisRow * imageWidth], imageWidth);
		Transform1D(&imageData[thisRow * imageWidth], imageWidth);
	}
	MyBarrier(myId, nThreads + 1);
	pthread_exit(NULL);
}

// Child thread to do inverse transform
void* iTransform2DThread(void* v) {
	long myId = (long)v;

	int rowsPerThread = imageHeight / nThreads;
	int startingRow = myId * rowsPerThread;

	for (int r = 0; r < rowsPerThread; ++r) {
		int thisRow = startingRow + r;
		iTransform1D(&imageData[thisRow * imageWidth], imageWidth);
		ReverseArray(&imageData[thisRow * imageWidth], imageWidth);
	}
	MyBarrier(myId, nThreads + 1);
	pthread_exit(NULL);
}

void Transform2D(const char* inputFN) 
{	// Do the 2D transform here.
	InputImage image(inputFN);  // Create the helper object for reading the image
	// Create the global pointer to the image array data
	// Create 16 threads
	// Wait for all threads complete
	// Write the transformed data
	imageWidth = image.GetWidth();
	imageHeight = image.GetHeight();
	imageData = image.GetImageData();

	// Precomputing the Weight array
	pComputeW(imageWidth);	

	// Initiate the Barrier
	MyBarrier_Init(nThreads + 1);

	pthread_t pt_1[nThreads];

	// Now start threads
	for (long i = 0; i < nThreads; ++i) {
		// Create the thread
		pthread_create(&pt_1[i], NULL, Transform2DThread, (void *)i);
	}
	
	// Waiting for all threads to finish their jobs 
	MyBarrier(nThreads, nThreads + 1);

	Transpose(imageData, imageWidth, imageHeight);

	pthread_t pt_2[nThreads];

	// Now start threads
	for (long i = 0; i < nThreads; ++i) {
		// Create the thread
		pthread_create(&pt_2[i], NULL, Transform2DThread, (void*)i);
	}

	MyBarrier(nThreads, nThreads + 1);

	Transpose(imageData, imageWidth, imageHeight);

	string fn("MyAfter2d.txt");
	image.SaveImageData(fn.c_str(), imageData, imageWidth, imageHeight);

	pthread_t pt_3[nThreads];

	// Now start threads
	for (long i = 0; i < nThreads; ++i) {
		// Create the thread
		pthread_create(&pt_3[i], NULL, iTransform2DThread, (void*)i);
	}

	MyBarrier(nThreads, nThreads + 1);

	Transpose(imageData, imageWidth, imageHeight);

	pthread_t pt_4[nThreads];

	// Now start threads
	for (long i = 0; i < nThreads; ++i) {
		// Create the thread
		pthread_create(&pt_4[i], NULL, iTransform2DThread, (void*)i);
	}

	MyBarrier(nThreads, nThreads + 1);

	Transpose(imageData, imageWidth, imageHeight);

	string fn_i("MyAfterInverse.txt");
	image.SaveImageData(fn_i.c_str(), imageData, imageWidth, imageHeight);
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  
  Transform2D(fn.c_str()); // Perform the transform.
}  
  

  

/*
* Copyright (c) 2008 Christian Buchner <Christian.Buchner@gmail.com>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY Christian Buchner ''AS IS'' AND ANY 
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Christian Buchner BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CUDACOMPLEX_H
#define CUDACOMPLEX_H

// Depending on whether we're running inside the CUDA compiler, define the __host_
// and __device__ intrinsics, otherwise just make the functions static to prevent
// linkage issues (duplicate symbols and such)
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define HOST static inline
#define DEVICE static inline
#define HOSTDEVICE static inline
#endif

// Struct alignment is handled differently between the CUDA compiler and other
// compilers (e.g. GCC, MS Visual C++ .NET)
#ifdef __CUDACC__
#define ALIGN(x)  __align__(x)
#else
#if defined(_MSC_VER) && (_MSC_VER >= 1300)
// Visual C++ .NET and later
#define ALIGN(x) __declspec(align(x)) 
#else
#if defined(__GNUC__)
// GCC
#define ALIGN(x)  __attribute__ ((aligned (x)))
#else
// all other compilers
#define ALIGN(x) 
#endif
#endif
#endif

// Somehow in emulation mode the code won't compile Mac OS X 1.1 CUDA SDK when the
// operators below make use of references (compiler bug?). So instead we compile
// the code to pass everything through the stack. Slower, but works.
// I am not sure how the Linux CUDA SDK will behave, so currently when I detect
// Microsoft's Visual C++.NET I always allow it to use references.
#if !defined(__DEVICE_EMULATION__) || (defined(_MSC_VER) && (_MSC_VER >= 1300))
#define REF(x) &x
#define ARRAYREF(x,y) (&x)[y]
#else
#define REF(x) x
#define ARRAYREF(x,y) x[y]
#endif

typedef struct ALIGN(8) _cudacomplex {
  double real;
  double img;
  
  // assignment of a scalar to complex
HOSTDEVICE _cudacomplex& operator=(const double REF(a)) {
     real = a; img = 0;
     return *this;
  };

  // assignment of a pair of doubles to complex
HOSTDEVICE _cudacomplex& operator=(const double ARRAYREF(a,2)) {
     real = a[0]; img = a[1];
     return *this;
  };

} cudacomplex;

// add complex numbers
HOSTDEVICE cudacomplex operator+(const cudacomplex REF(a), const cudacomplex REF(b)) {
   cudacomplex result = { a.real + b.real, a.img  + b.img };
   return result;
}

// add scalar to complex
HOSTDEVICE cudacomplex operator+(const cudacomplex REF(a), const double REF(b)) {
   cudacomplex result = { a.real + b, a.img };
   return result;
}

// add complex to scalar
HOSTDEVICE cudacomplex operator+(const double REF(a), const cudacomplex REF(b)) {
   cudacomplex result = { a + b.real, b.img };
   return result;
}

// subtract complex numbers
HOSTDEVICE cudacomplex operator-(const cudacomplex REF(a), const cudacomplex REF(b)) {
   cudacomplex result = { a.real - b.real, a.img  - b.img };
   return result;
}

// subtract scalar from complex
HOSTDEVICE cudacomplex operator-(const cudacomplex REF(a), const double REF(b)) {
   cudacomplex result = { a.real - b, a.img };
   return result;
}

// subtract complex from scalar
HOSTDEVICE cudacomplex operator-(const double REF(a), const cudacomplex REF(b)) {
   cudacomplex result = { a - b.real, -b.img };
   return result;
}

// multiply complex numbers
HOSTDEVICE cudacomplex operator*(const cudacomplex REF(a), const cudacomplex REF(b)) {
   cudacomplex result = { a.real * b.real - a.img  * b.img,
                      a.img  * b.real + a.real * b.img };
   return result;
}

// multiply complex with scalar
HOSTDEVICE cudacomplex operator*(const cudacomplex REF(a), const double REF(b)) {
   cudacomplex result = { a.real * b, a.img  * b };
   return result;
}

// multiply scalar with complex
HOSTDEVICE cudacomplex operator*(const double REF(a), const cudacomplex REF(b)) {
   cudacomplex result = { a * b.real, a * b.img };
   return result;
}

// divide complex numbers
HOSTDEVICE cudacomplex operator/(const cudacomplex REF(a), const cudacomplex REF(b)) {
   double tmp   = ( b.real * b.real + b.img * b.img );
   cudacomplex result = { (a.real * b.real + a.img  * b.img ) / tmp,
                      (a.img  * b.real - a.real * b.img ) / tmp };
   return result;
}

// divide complex by scalar
HOSTDEVICE cudacomplex operator/(const cudacomplex REF(a), const double REF(b)) {
   cudacomplex result = { a.real / b, a.img  / b };
   return result;
}

// divide scalar by complex
HOSTDEVICE cudacomplex operator/(const double REF(a), const cudacomplex REF(b)) {
   double tmp   = ( b.real * b.real + b.img * b.img );
   cudacomplex result = { ( a * b.real ) / tmp, ( -a * b.img ) / tmp };
   return result;
}

// complex conjugate
HOSTDEVICE cudacomplex operator~(const cudacomplex REF(a)) {
   cudacomplex result = { a.real, -a.img };
   return result;
}

#endif // #ifndef CUDACOMPLEX_H

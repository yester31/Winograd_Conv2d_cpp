# Winograd_Conv2d_cpp
Winograd Convolution Implementation

## Enviroments
* Windows 10 laptop
* CPU 11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz (cpu)

##  Winograd Convolution
- WinogradConvolutionFunction.cpp
- Minimum filtering algorithm of finite impulse response filter.
- Time check(kernel[1, 3, 3, 3])
	- input[1, 3, 512, 512]
		- dur_time(Naive Convolution) = 6.07500[msec]
		- dur_time(Winograd Convolution) = 1.93700[msec]
	- input[1, 3, 1024, 1024]
		- dur_time(Naive Convolution) = 26.18000[msec]
		- dur_time(Winograd Convolution) = 7.77900[msec]
	- input[1, 3, 2048, 2048]
		- dur_time(Naive Convolution) = 84.76600[msec]
		- dur_time(Winograd Convolution) = 29.81900[msec]
	- input[1, 3, 4096, 4096]
		- dur_time(Naive Convolution) = 341.64801[msec]
		- dur_time(Winograd Convolution) = 113.30500[msec]

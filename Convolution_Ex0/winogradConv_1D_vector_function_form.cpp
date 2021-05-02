#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <chrono>

using namespace std;
using namespace chrono;
using namespace cv;


/***************************************************************************
	filterTransform 함수,  V
*****************************************************************************/

void filterTransform(vector<float>& Output_U, vector<float> filter, int output_c, int input_c) {

	int temp1u = input_c * 16;
	int temp1f = input_c * 9;
	for (int ⁠n_idx = 0; ⁠n_idx < output_c; ⁠n_idx++)
	{
		int temp2u = ⁠n_idx * temp1u;
		int temp2f = ⁠n_idx * temp1f;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int u_idx = ⁠c_idx * 16 + temp2u;
			int f_idx = ⁠c_idx * 9 + temp2f;

			float* g1 = &filter[f_idx];
			float* g2 = &filter[f_idx + 1];
			float* g3 = &filter[f_idx + 2];
			float* g4 = &filter[f_idx + 3];
			float* g5 = &filter[f_idx + 4];
			float* g6 = &filter[f_idx + 5];
			float* g7 = &filter[f_idx + 6];
			float* g8 = &filter[f_idx + 7];
			float* g9 = &filter[f_idx + 8];

			Output_U[u_idx] = *g1;
			Output_U[u_idx + 1] = (*g1 + *g2 + *g3) / 2.f;
			Output_U[u_idx + 2] = (*g1 - *g2 + *g3) / 2.f;
			Output_U[u_idx + 3] = *g3;

			float temp1 = *g1 + *g4 + *g7;
			float temp2 = *g2 + *g5 + *g8;
			float temp3 = *g3 + *g6 + *g9;

			Output_U[u_idx + 4] = (temp1) / 2.f;
			Output_U[u_idx + 5] = (temp1 + temp2 + temp3) / 4.f;
			Output_U[u_idx + 6] = (temp1 - temp2 + temp3) / 4.f;
			Output_U[u_idx + 7] = (temp3) / 2.f;

			float temp4 = *g1 - *g4 + *g7;
			float temp5 = *g2 - *g5 + *g8;
			float temp6 = *g3 - *g6 + *g9;

			Output_U[u_idx + 8] = (temp4) / 2.f;
			Output_U[u_idx + 9] = (temp4 + temp5 + temp6) / 4.f;
			Output_U[u_idx + 10] = (temp4 - temp5 + temp6) / 4.f;
			Output_U[u_idx + 11] = (temp6) / 2.f;

			Output_U[u_idx + 12] = *g7;
			Output_U[u_idx + 13] = (*g7 + *g8 + *g9) / 2.f;
			Output_U[u_idx + 14] = (*g7 - *g8 + *g9) / 2.f;
			Output_U[u_idx + 15] = *g9;

		}
	}
}

void valueCheck(vector<float>& valueCheckInput, int input_n, int input_c, int input_h, int input_w, int offset = 0) {
	if (offset == 1) { input_n = 1; }

	int temp1 = input_w * input_h * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int g_idx = w_idx + temp4;
					cout << setw(5) << valueCheckInput[g_idx] << " ";
				}cout << endl;
			}cout << endl; cout << endl;
		}cout << endl;
	}cout << endl;
}

void convolution(vector<float> &convOutput, vector<float> &convInput, vector<float> &kernel, int kernelSize, int stride, int input_n, int input_c, int input_h, int input_w, int ouput_c) {
	int outputHeightSize = ((input_h - kernelSize) / stride) + 1;
	int outputWidthSize = ((input_w - kernelSize) / stride) + 1;
	//Conv_output.resize(input_n * Ouput_C * outputHeightSize * outputHeightSize);
	//cout << "===== Convolution ===== \n";

	int temp1i = input_h * input_w *input_c;
	int temp1o = outputHeightSize * outputWidthSize * ouput_c;
	int temp1k = kernelSize * kernelSize * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2i = ⁠n_idx * temp1i;
		int temp2o = ⁠n_idx * temp1o;
		for (int k_idx = 0; k_idx < ouput_c; k_idx++)
		{
			int temp2k = k_idx * temp1k;
			int temp3o = k_idx * outputHeightSize * outputWidthSize + temp2o;
			for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
			{
				int temp3i = ⁠c_idx * input_w * input_h + temp2i;
				int temp3k = ⁠c_idx * kernelSize * kernelSize + temp2k;
				for (int rowStride = 0; rowStride < outputHeightSize; rowStride++) {
					int temp4o = rowStride * outputWidthSize + temp3o;
					for (int colStride = 0; colStride < outputWidthSize; colStride++) {
						float sum = 0;
						int g_idx_o = colStride + temp4o;
						for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
							int temp4i = x * input_w + temp3i;
							int temp4k = (x - rowStride * stride) * kernelSize + temp3k;
							for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
								int ⁠g_idx_i = y + temp4i;
								int g_idx_k = (y - colStride * stride) + temp4k;
								sum += convInput[⁠g_idx_i] * kernel[g_idx_k];
							}
						}
						convOutput[g_idx_o] += sum;
					}
				}
			}
		}
	}
}

void winogradConv2d(vector<float> &convOutput, vector<float> &convInput, vector<float> kernel, int input_n, int input_c, int input_h, int input_w, int output_c) {

	vector<float> U(output_c * input_c * 16);
	// filterTransform 수행
	filterTransform(U, kernel, output_c, input_c);

	int output_h = input_h - 2;
	int output_w = input_w - 2;
	int temp_o = output_c * output_h * output_w;
	int temp_i = input_c * input_h * input_w;
	int temp_u = input_c * 16;
	for (int n = 0; n < input_n; n++)
	{
		int temp_i2 = n * temp_i;
		int temp_o2 = n * temp_o;

		for (int row = 0; row < output_h; row += 2)
		{
			int row_idx1 = row * input_w;
			int row_idx2 = row_idx1 + input_w;
			int row_idx3 = row_idx2 + input_w;
			int row_idx4 = row_idx3 + input_w;

			int row_idxo1 = row * output_w;
			int row_idxo2 = row_idxo1 + output_w;

			for (int col = 0; col < output_w; col += 2)
			{
				for (int outch = 0; outch < output_c; outch++)
				{
					int temp_u2 = outch * temp_u;
					int ot_idx1 = outch * output_h * output_w + temp_o2 + col + row_idxo1;
					int ot_idx3 = outch * output_h * output_w + temp_o2 + col + row_idxo2;

					float y1 = 0;
					float y2 = 0;
					float y3 = 0;
					float y4 = 0;

					for (int inch = 0; inch < input_c; inch++)
					{
						int temp_ic = inch * input_h * input_w;
						int u_idx = inch * 16 + temp_u2; // U idex

						int t_idx1 = temp_ic + temp_i2 + row_idx1 + col;
						int t_idx2 = temp_ic + temp_i2 + row_idx2 + col;
						int t_idx3 = temp_ic + temp_i2 + row_idx3 + col;
						int t_idx4 = temp_ic + temp_i2 + row_idx4 + col;

						float* d1 = &convInput[t_idx1];
						float* d2 = &convInput[t_idx1 + 1];
						float* d3 = &convInput[t_idx1 + 2];
						float* d4 = &convInput[t_idx1 + 3];

						float* d5 = &convInput[t_idx2];
						float* d6 = &convInput[t_idx2 + 1];
						float* d7 = &convInput[t_idx2 + 2];
						float* d8 = &convInput[t_idx2 + 3];

						float* d9 = &convInput[t_idx3];
						float* d10 = &convInput[t_idx3 + 1];
						float* d11 = &convInput[t_idx3 + 2];
						float* d12 = &convInput[t_idx3 + 3];

						float* d13 = &convInput[t_idx4];
						float* d14 = &convInput[t_idx4 + 1];
						float* d15 = &convInput[t_idx4 + 2];
						float* d16 = &convInput[t_idx4 + 3];

						float dd1 = *d11 - (*d3);
						float dd2 = *d2 - (*d10);
						float dd3 = *d7 + (*d11);
						float dd4 = *d6 + (*d10);
						float dd5 = *d7 - (*d11);
						float dd6 = *d10 - (*d6);
						float dd7 = *d15 - (*d7);
						float dd8 = *d6 - (*d14);

						float v1 = *d1 - *d9 + dd1;
						float v2 = dd2 - dd1;//
						float v3 = -dd1 - dd2;//
						float v4 = dd2 - *d4 + *d12;

						float v5 = *d5 + *d9 - dd3;
						float v6 = dd4 + dd3;
						float v7 = dd3 - dd4;
						float v8 = dd4 - *d8 - *d12;

						float v9 = *d9 - *d5 + dd5;
						float v10 = dd6 - dd5;
						float v11 = -(dd6 + dd5);
						float v12 = dd6 + *d8 - *d12;

						float v13 = *d5 - *d13 + dd7;
						float v14 = dd8 - dd7;
						float v15 = -dd7 - dd8;
						float v16 = dd8 - *d8 + *d16;

						// U . V
						float m1 = v1 * U[u_idx];
						float m2 = v2 * U[u_idx + 1];
						float m3 = v3 * U[u_idx + 2];
						float m4 = v4 * U[u_idx + 3];
						float m5 = v5 * U[u_idx + 4];
						float m6 = v6 * U[u_idx + 5];
						float m7 = v7 * U[u_idx + 6];
						float m8 = v8 * U[u_idx + 7];
						float m9 = v9 * U[u_idx + 8];
						float m10 = v10 * U[u_idx + 9];
						float m11 = v11 * U[u_idx + 10];
						float m12 = v12 * U[u_idx + 11];
						float m13 = v13 * U[u_idx + 12];
						float m14 = v14 * U[u_idx + 13];
						float m15 = v15 * U[u_idx + 14];
						float m16 = v16 * U[u_idx + 15];

						// output transfom
						float sub_y1 = m2 + m6 + m10;
						float sub_y2 = m3 + m7 + m11;
						float sub_y3 = m6 - m10 - m14;
						float sub_y4 = m7 - m11 - m15;

						y1 += m1 + m5 + m9 + sub_y1 + sub_y2;
						y2 += sub_y1 - sub_y2 - m4 - m8 - m12;
						y3 += m5 - m9 - m13 + sub_y3 + sub_y4;
						y4 += sub_y3 - sub_y4 - m8 + m12 + m16;
					}

					convOutput[ot_idx1] = y1;
					convOutput[ot_idx1 + 1] = y2;
					convOutput[ot_idx3] = y3;
					convOutput[ot_idx3 + 1] = y4;
				}
			}
		}
	}



}

int main()
{
	cout << "Winograd Convolutions function! \n\n";

	int batchSize = 1;
	int input_n = batchSize;
	int input_c = 3;
	int input_h = 32;
	int input_w = 32;

	// input[input_n][input_c][input_h][input_w] 
	// 임시 input 값 생성 1++
	vector<float> input(input_n * input_c * input_h * input_w);
	float count = 1.f;
	int temp1 = input_c * input_h * input_w;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_h * input_w + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int g_idx = w_idx + temp4;
					input[g_idx] = count;
					count++;
				}
			}
		}
	}

	int output_c = 4;
	// h[output_c][input_c][KernelSize_h][KernelSize_w] 
	// h[4][3][3][3]
	// 임시 filter 값 생성 1
	vector<float> h(output_c * input_c * 3 * 3);
	//float count = 1.f;
	int temp1h = input_c * 3 * 3;
	for (int ⁠n_idx = 0; ⁠n_idx < output_c; ⁠n_idx++)
	{
		int temp2h = ⁠n_idx * temp1h;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3h = ⁠c_idx * 3 * 3 + temp2h;
			for (int ⁠h_idx = 0; ⁠h_idx < 3; ⁠h_idx++)
			{
				int temp4h = ⁠h_idx * 3 + temp3h;
				for (int w_idx = 0; w_idx < 3; w_idx++)
				{
					int g_idxh = w_idx + temp4h;
					h[g_idxh] = 1.f;
					//h[g_idx] = count;
					//count++;
				}
			}
		}
	}

	// h(filater) 값 체크
	valueCheck(h, output_c, input_c, 3, 3);
	// input 값 체크
	valueCheck(input, input_n, input_c, input_h, input_w);

	int output_h = input_h - 2;
	int output_w = input_w - 2;
	// output[input_n][output_c][output_h][output_w] 
	vector<float> output(input_n * output_c * output_h * output_w);

	long long start_usec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

	winogradConv2d(output, input, h, input_n, input_c, input_h, input_w, output_c);

	long long end_usec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	int frame_sec = int(end_usec - start_usec);

	// output 값 체크
	cout << "===== Winograd Convolution ===== \n";
	valueCheck(output, input_n, output_c, output_h, output_w);
	cout << frame_sec << "u sec (Winograd Convolution)" << endl;
	return 0;
}
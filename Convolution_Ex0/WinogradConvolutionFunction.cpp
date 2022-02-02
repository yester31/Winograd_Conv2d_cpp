#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <time.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

/***************************************************************************
	comparing winograd and naive Convolution 
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

void zeroPadding(vector<float> &zeroPaddingOutput, vector<float> &zeroPaddingInput, int input_n, int input_c, int input_h, int input_w, int pad_l, int pad_r, int pad_t, int pad_b) {

	int temp1 = input_w * input_h * input_c;
	int temp1o = (input_h + pad_t + pad_b)*(input_w + pad_l + pad_r)* input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		int temp2o = ⁠n_idx * temp1o;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			int temp3o = ⁠c_idx * (input_w + pad_l + pad_r) * (input_h + pad_t + pad_b) + temp2o;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				int temp4o = (⁠h_idx + pad_t)*(input_w + pad_l + pad_r) + pad_l + temp3o;

				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int ⁠g_idx = w_idx + temp4;
					int g_idx_Output = w_idx + temp4o;
					zeroPaddingOutput[g_idx_Output] = zeroPaddingInput[⁠g_idx];
				}
			}
		}
	}
}

void winogradConv2d(vector<float> &convOutput, vector<float> &convInput, vector<float> kernel, int input_n, int input_c, int input_h, int input_w, int output_c) {
	printf("===== Winograd Convolution ===== \n");

	// 1. filterTransform 수행
	vector<float> U(output_c * input_c * 16);
	filterTransform(U, kernel, output_c, input_c);

	// 2. input transform 및 element wise multiplication, output transform수행
	int output_h_z = input_h - 2;
	int output_w_z = input_w - 2;

	int temp_o = output_c * output_h_z * output_w_z;
	int temp_i = input_c * input_h * input_w;
	int temp_u = input_c * 16;

	// 3. Element wise matrix multiplication
	for (int n = 0; n < input_n; n++){
		int temp_i2 = n * temp_i;
		int temp_o2 = n * temp_o;

		for (int row = 0; row < output_h_z; row += 2){
			int row_idx1 = row * input_w;
			int row_idx2 = row_idx1 + input_w;
			int row_idx3 = row_idx2 + input_w;
			int row_idx4 = row_idx3 + input_w;
			int row_idxo1 = row * output_w_z;
			int row_idxo2 = row_idxo1 + output_w_z;

			for (int col = 0; col < output_w_z; col += 2){

				for (int outch = 0; outch < output_c; outch++){
					int temp_u2 = outch * temp_u;
					int ot_idx1 = outch * output_h_z * output_w_z + temp_o2 + col + row_idxo1;
					int ot_idx3 = outch * output_h_z * output_w_z + temp_o2 + col + row_idxo2;
					float y1 = 0, y2 = 0, y3 = 0,y4 = 0;

					for (int inch = 0; inch < input_c; inch++){
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

						// 4. output transfom
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

void convolution(float* output, float* input, float* weight, int IN, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW) {
	printf("===== Conventional Convolution ===== \n");

	int OH = ((IH - KH) / SH) + 1;
	int OW = ((IW - KW) / SW) + 1;

	int C_offset_i, C_offset_o, C_offset_k, H_offset_i, H_offset_o, H_offset_k, W_offset_i, W_offset_o, W_offset_k, ⁠g_idx_i, g_idx_o, g_idx_k;
	int N_offset_i = IC * IH * IW;
	int N_offset_o = OC * OH * OW;
	int N_offset_k = IC * KH * KW;

	for (int ⁠n_idx = 0; ⁠n_idx < IN; ⁠n_idx++) {
		C_offset_i = ⁠n_idx * N_offset_i;
		C_offset_o = ⁠n_idx * N_offset_o;

		for (int k_idx = 0; k_idx < OC; k_idx++) {
			C_offset_k = k_idx * N_offset_k;
			H_offset_o = k_idx * OH * OW + C_offset_o;

			for (int ⁠c_idx = 0; ⁠c_idx < IC; ⁠c_idx++) {
				H_offset_i = ⁠c_idx * IH * IW + C_offset_i;
				H_offset_k = ⁠c_idx * KH * KW + C_offset_k;

				for (int rowStride = 0; rowStride < OH; rowStride++) {
					W_offset_o = rowStride * OW + H_offset_o;

					for (int colStride = 0; colStride < OW; colStride++) {
						float sum = 0;
						g_idx_o = colStride + W_offset_o;

						for (int y = rowStride * SH; y < rowStride * SH + KH; y++) {
							W_offset_i = y * IW + H_offset_i;
							W_offset_k = (y - rowStride * SH) * KH + H_offset_k;

							for (int x = colStride * SW; x < colStride * SW + KW; x++) {

								⁠g_idx_i = x + W_offset_i;
								g_idx_k = (x - colStride * SW) + W_offset_k;
								sum += input[⁠g_idx_i] * weight[g_idx_k];
							}
						}
						output[g_idx_o] += sum;
					}
				}
			}
		}
	}
}


// 결과값 비교
void compareResults(float *result_1, float *result_2, int size) {
	bool result = true;
	for (int i = 0; i < size; i++) {
		if ((result_1[i]) != result_2[i]) {
			printf("[%d] The results is not matched! (%f, %f)\n", i, result_1[i], result_2[i]);
			result = false;
		}
	}
	if (result)printf("Results is same!! works well! \n");
	else printf("results is not matched! \n");
}

// 데이터 초기화(스칼라 값) Default = 1
void initDataScalar(float* ptr, unsigned int size, float tt = 1) {
	while (size--) {
		*ptr++ = tt;
	}
}

// 데이터 초기화(1부터 1씩 증가)
void initDataStep(float* ptr, unsigned int size) {
	float tt = 1;
	while (size--) {
		*ptr++ = tt++;
	}
}

// 데이터 초기화(랜덤 값)
void initDataRandom(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = rand() % 10;
	}
}

int main()
{
	cout << "Winograd & Naive Convolutions function! \n\n";
	// *** notice *** 
	// In case of winograd, Only use stride = 1, kernel = 3 and even number input size.

	// 0) parameter setting
	int input_n = 1;
	int input_c = 3;
	int input_h = 512;
	int input_w = 512;
	int stride = 1;
	int kernel_size = 3;
	int output_c = 1;
	int pad_l = 0;
	int pad_r = 0;
	int pad_t = 0;
	int pad_b = 0;
	int output_p = ((input_h + pad_t + pad_b - kernel_size) / stride) + 1;
	int output_q = ((input_w + pad_l + pad_r - kernel_size) / stride) + 1;


	// 1) generation input data
	// input[input_n][input_c][input_h][input_w] 
	vector<float> input(input_n * input_c * input_h * input_w);
	initDataRandom(input.data(), input.size());
	// input value check
	//valueCheck(input, input_n, input_c, input_h, input_w);


	// 2) generation filter value
	// h[output_c][input_c][kernel_size][kernel_size] 
	vector<float> h(output_c * input_c * kernel_size * kernel_size);
	initDataRandom(h.data(), h.size());
	// h(filter) value check
	//valueCheck(h, output_c, input_c, 3, 3);


	// 3) assignment output space 
	// output[input_n][output_c][output_p][output_q] 
	vector<float> wino_output(input_n * output_c * output_p * output_q);
	vector<float> naive_output(input_n * output_c * output_p * output_q);

	vector<float> input_padded(input_n * input_c * (input_h + pad_t + pad_b) * (input_w + pad_l + pad_r));
	zeroPadding(input_padded, input, input_n, input_c, input_h, input_w, pad_t, pad_b, pad_l, pad_r);

	// 4) execute winograd convolution
	cout << "===== Winograd Convolution ===== \n";
	long long start_usec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	winogradConv2d(wino_output, input_padded, h, input_n, input_c, input_h + pad_t + pad_b, input_w + pad_l + pad_r, output_c);
	long long end_usec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	// Winograd output value check
	//valueCheck(wino_output, input_n, output_c, output_p, output_q);


	// 5) execute naive convolution
	cout << "===== Naive Convolution ===== \n";
	long long start_usec2 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

	//valueCheck(inputPadding, input_n, input_c, input_h + pad_t + pad_b, input_w + pad_l + pad_r);
	convolution(naive_output.data(), input_padded.data(), h.data(), input_n, input_c, input_h + pad_t + pad_b, input_w + pad_l + pad_r, output_c, kernel_size, kernel_size, stride, stride);
	long long end_usec2 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	// Naive output value check
	//valueCheck(naive_output, input_n, output_c, output_p, output_q);


	// 6) print results
	compareResults(naive_output.data(), wino_output.data(), wino_output.size());
	printf("input[%4d,%4d,%4d,%4d] kernel[%4d,%4d,%4d,%4d] output[%4d,%4d,%4d,%4d]\n\n", 
		input_n, input_c, input_h, input_w, output_c, input_c, kernel_size, kernel_size, input_n, output_c, output_p, output_q);
	printf("dur_time(Naive Convolution)     = %6.5f [msec] \n", (end_usec2 - start_usec2) / 1000.f);
	printf("dur_time(Winograd Convolution)  = %6.5f [msec] \n", (end_usec - start_usec) / 1000.f);


	return 0;
}

//input[1, 3, 512, 512] kernel[1, 3, 3, 3] output[1, 1, 510, 510]
//dur_time(Naive Convolution) = 6.07500[msec]
//dur_time(Winograd Convolution) = 1.93700[msec]

//input[   1,   3,1024,1024] kernel[   1,   3,   3,   3] output[   1,   1,1022,1022]
//dur_time(Naive Convolution) = 26.18000[msec]
//dur_time(Winograd Convolution) = 7.77900[msec]

//input[1, 3, 2048, 2048] kernel[1, 3, 3, 3] output[1, 1, 2046, 2046]
//dur_time(Naive Convolution) = 84.76600[msec]
//dur_time(Winograd Convolution) = 29.81900[msec]

//input[1, 3, 4096, 4096] kernel[1, 3, 3, 3] output[1, 1, 4094, 4094]
//dur_time(Naive Convolution) = 341.64801[msec]
//dur_time(Winograd Convolution) = 113.30500[msec]
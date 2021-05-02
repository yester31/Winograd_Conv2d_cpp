#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"
#include <time.h>

using namespace cv;
using namespace std;

/***************************************************************************
	filterTransform 함수,  V 만들려다가 ... 다함.... -> 최적화 및 함수 형태
*****************************************************************************/

void filterTransform(vector<float> & Output_U, vector<float> filter, int out_ch, int in_ch) {

	int temp1u = in_ch * 4 * 4;
	int temp1f = in_ch * 3 * 3;
	for (int ⁠n_idx = 0; ⁠n_idx < out_ch; ⁠n_idx++)
	{
		int temp2u = ⁠n_idx * temp1u;
		int temp2f = ⁠n_idx * temp1f;
		for (int ⁠c_idx = 0; ⁠c_idx < in_ch; ⁠c_idx++)
		{
			int u_idx = ⁠c_idx * 4 * 4 + temp2u;
			int f_idx = ⁠c_idx * 3 * 3 + temp2f;

			float *g1 = &filter[f_idx];
			float *g2 = &filter[f_idx + 1];
			float *g3 = &filter[f_idx + 2];
			float *g4 = &filter[f_idx + 3];
			float *g5 = &filter[f_idx + 4];
			float *g6 = &filter[f_idx + 5];
			float *g7 = &filter[f_idx + 6];
			float *g8 = &filter[f_idx + 7];
			float *g9 = &filter[f_idx + 8];

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

void valueCheck(vector<float> &valueCheckInput, int input_n, int input_c, int input_h, int input_w, int offset = 0) {
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
		}
	}
}

int main()
{
	cout << "Winograd Convolutions filterTransform function! \n\n";


	int FeatureCh = 3;
	int out_ch = 4;

	// h[O_Ch][ln_Ch][KernelSize_h][KernelSize_w] 
	// h[4][3][3][3]
	// 임시 filter 값 
	vector<float> h(out_ch*FeatureCh * 3 * 3);
	//float count = 1.f;
	int temp1h = FeatureCh * 3 * 3;
	for (int ⁠n_idx = 0; ⁠n_idx < out_ch; ⁠n_idx++)
	{
		int temp2h = ⁠n_idx * temp1h;
		for (int ⁠c_idx = 0; ⁠c_idx < FeatureCh; ⁠c_idx++)
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

	// filater 값 체크
	valueCheck(h, out_ch, FeatureCh, 3, 3);

	// U[O_Ch][ln_Ch][4][4] 
	vector<float> U(out_ch * FeatureCh * 4 * 4);

	// filterTransform 수행
	filterTransform(U, h, out_ch, FeatureCh);


	int batch_size = 1;
	//int FeatureCh = 3;
	int input_H = 10;
	int input_W = 10;
	int area_i = input_H * input_W;
	//int out_ch = 4;

	// d[N][IC][input_H][input_W] 
	// d[1][3][10][10]
	// 임시 input 값 

	vector<float> input(batch_size * FeatureCh * area_i);
	float count = 1.f;
	int temp1 = FeatureCh * input_H * input_W;
	for (int ⁠n_idx = 0; ⁠n_idx < batch_size; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < FeatureCh; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_H * input_W + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_H; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_W + temp3;
				for (int w_idx = 0; w_idx < input_W; w_idx++)
				{
					int g_idx = w_idx + temp4;
					input[g_idx] = count;
					count++;
				}
			}
		}
	}


	// filater 값 체크
	valueCheck(input, batch_size, FeatureCh, input_H, input_W);

	// output[N][out_ch][4][4] 
	int output_H = input_H - 2;
	int output_W = input_W - 2;
	int area_o = output_H * output_W;
	vector<float> output(batch_size * out_ch * area_o);


	int temp_o = out_ch * area_o;
	int temp_i = FeatureCh * area_i;
	int temp_u = FeatureCh * 16;

	for (int n = 0; n < batch_size; n++)
	{
		int temp_i2 = n * temp_i;
		int temp_o2 = n * temp_o;

		for (int row = 0; row < output_H; row += 2)
		{
			int row_idx1 = row * input_W;
			int row_idx2 = row_idx1 + input_W;
			int row_idx3 = row_idx2 + input_W;
			int row_idx4 = row_idx3 + input_W;

			int row_idxo1 = row * output_W;
			int row_idxo2 = row_idxo1 + output_W;

			for (int col = 0; col < output_W; col += 2)
			{			
				for (int outch = 0; outch < out_ch; outch++)
				{
					int temp_u2  = outch * temp_u;
					int temp_ot1 = outch * area_o + temp_o2 + col + row_idxo1;
					int temp_ot3 = outch * area_o + temp_o2 + col + row_idxo2;

					float y1 = 0;
					float y2 = 0;
					float y3 = 0;
					float y4 = 0;

					for (int inch = 0; inch < FeatureCh; inch++)
					{
						int temp_ic = inch * area_i;
						int u_idx = inch * 16 + temp_u2; // U idex
						int subidx = temp_ic + temp_i2 + col;
						int t_idx1 = subidx + row_idx1;
						int t_idx2 = subidx + row_idx2;
						int t_idx3 = subidx + row_idx3;
						int t_idx4 = subidx + row_idx4;

						float *d1 = &input[t_idx1];
						float *d2 = &input[t_idx1 + 1];
						float *d3 = &input[t_idx1 + 2];
						float *d4 = &input[t_idx1 + 3];

						float *d5 = &input[t_idx2];
						float *d6 = &input[t_idx2 + 1];
						float *d7 = &input[t_idx2 + 2];
						float *d8 = &input[t_idx2 + 3];

						float *d9 = &input[t_idx3];
						float *d10 = &input[t_idx3 + 1];
						float *d11 = &input[t_idx3 + 2];
						float *d12 = &input[t_idx3 + 3];

						float *d13 = &input[t_idx4];
						float *d14 = &input[t_idx4 + 1];
						float *d15 = &input[t_idx4 + 2];
						float *d16 = &input[t_idx4 + 3];

						float dd1 =  *d11 - (*d3);
						float dd2 =  *d2 - (*d10);
						float dd3 =  *d7 + (*d11);
						float dd4 =  *d6 + (*d10);
						float dd5 =  *d7 - (*d11);
						float dd6 = *d10 - (*d6);
						float dd7 = *d15 - (*d7);
						float dd8 =  *d6 - (*d14);

						// U . V
						float m1 = (*d1 - *d9 + dd1) * U[u_idx];
						float m2 = (dd2 - dd1) * U[u_idx + 1];
						float m3 = (-dd1 - dd2) * U[u_idx + 2];
						float m4 = (dd2 - *d4 + *d12) * U[u_idx + 3];

						float m5 = (*d5 + *d9 - dd3) * U[u_idx + 4];
						float m6 = (dd4 + dd3) * U[u_idx + 5];
						float m7 = (dd3 - dd4) * U[u_idx + 6];
						float m8 = (dd4 - *d8 - *d12) * U[u_idx + 7];

						float m9 = (*d9 - *d5 + dd5) * U[u_idx + 8];
						float m10 = (dd6 - dd5) * U[u_idx + 9];
						float m11 = (-dd6 - dd5) * U[u_idx + 10];
						float m12 = (dd6 + *d8 - *d12) * U[u_idx + 11];

						float m13 = (*d5 - *d13 + dd7) * U[u_idx + 12];
						float m14 = (dd8 - dd7) * U[u_idx + 13];
						float m15 = (-dd7 - dd8) * U[u_idx + 14];
						float m16 = (dd8 - *d8 + *d16) * U[u_idx + 15];

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

					output[temp_ot1] = y1;
					output[temp_ot1 + 1] = y2;
					output[temp_ot3] = y3;
					output[temp_ot3 + 1] = y4;
				}
			}
		}
	}


	// U 값 체크
	valueCheck(output, batch_size, out_ch, input_H - 2, input_W - 2);




	return 0;
}
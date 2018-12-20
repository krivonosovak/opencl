__kernel void gpu_convolution_gmem(__global float * input, __global float * mask,
                                              __global float * output, int mask_width, int width)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    float res = 0.0;
    for (int j = 0; j < mask_width; ++j)
        for (int k = 0; k < mask_width; ++k)
        {
            int input_idx = (idx + j - mask_width / 2);
            int input_idy = (idy + k - mask_width / 2);

            if (input_idx >= 0 && input_idx < width && input_idy < width && input_idx >= 0)
                res += input[input_idx * width + input_idy] * mask[j * mask_width + k];
        }
    output[idx * width + idy] = res;
}


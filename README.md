# Medical image denoising using convolutional denoising autoencoders
#  

Brief:  
Project uses visual comparisions mainly based on DX and all-MIAS dataset, comparing outputs with CNN Autoencoder results, BM3D and NL means denoising algorithm  
Comparisions are based on structural similarity index measure(SSIM) instead of peak signal to noise ratio (PSNR) for its consistency and accuracy.   
A composite index of three measures, SSIM estimates the visual effects of shifts in image luminance, contrast and other remaining errors, collectively called structural changes.  
  
*Different Gaussian noise level are used to add noise to the image.  
*Datasets DX and MIAS are used as default datasets  
*They should be placed in data folder:  
*/data/dx  
*/data/all-mias 
  
  Multiple samples of different outputs can be found in Samples folder
  
For more information how to run please use main.py -h
  
Project based on Medical image denoising using convolutional denoising autoencoders by Lovedeep Gondara  
Paper: https://arxiv.org/pdf/1608.04667.pdf  
  
By Adam Mahameed  

Sample:
![Image of Output](https://github.com/adam-mah/Medical-Image-Denoising/blob/master/Samples/0.1%200%201%20(3).png?raw=true)  
  
  This code is under MIT License  
  Copyright (c) 2020 Adam Mahameed

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
  
  

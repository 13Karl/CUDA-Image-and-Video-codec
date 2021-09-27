# CUDA-Image-and-Video-codec

Codec created during my Ph.D. that processes image and video files using Nvidia GPUs. It is based on the JPEG2000 framework and includes the same features, although it is not compliant with it. To run the software, the requirements are as follows:

- **Windows:** CUDA 10+ as well as Microsoft Visual Studio 2019+. For the development, I used Visual Studio Community 2019. The new version, 2021, is expected to be fully compatible with the project.

- **Linux:** GCC 8+ and CUDA 10+ and CMake 3.10.2.  

As per hardware requirements, a **Nvidia GPU is required**. The ones tested are from series 1000, 2000 and 3000, using 6.1, 7.5 and 8.6 CC respectively. We do not guarantee compatibility with older GPUs. 

The guide is divided now in two different sections, one for Windows and another for Linux, with specific instructions on how-to compile. The run examples are provided after the compilation guide as they are the same regardless of the OS used.

## Windows guide

First of all, all these instructions correspond to the Visual Studio Community 2019 version. Maybe in future versions, some or all of this steps will not be necessary. 

We will first download the git repository and open it. There, before opening the CUDA_ImCod.sln file, you will need to navigate to CUDA_ImCod folder and update the file called CUDA_ImCod.vcxproj via your preferred editor, more specifically the sections where the CUDA props and targets are addressed:

- Line 120: \<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.2.props" />

- Line 236: \<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.2.targets" />

Replacing 10.2 with the CUDA version you have installed in your system (at the time of writing this guide, the latest version is 11.4).

Now is the time to open the CUDA_ImCod.sln file. Visual studio will popup, and, depending on the version used, it could prompt to update the solution to a newer version. Just accept the suggestion. Once you are in the solution, right-click on the CUDA_ImCod solution and click on properties.

In the properties window, navigate to the **“VC++ Directories”** and modify the **“Include Directories”**, changing the **“cub”** folder (or appending it to the end if the cub folder is not there already) to the one where you have your cub implementation (https://nvlabs.github.io/cub/). It is included as well within the source code of this project.

Next, under the CUDA C/C++ properties, update the **“Additional Include Directories”** section with the right information on where the cub folder is. On this same CUDA C/C++ section, select **“Device”** and change the **“compute_61,sm_61”** to the appropriate values depending on the gpu, i.e.:

- 1050, 1060, 1070, 1080: compute_61,sm_61
- 2060, 2070, 2080: compute_75,sm_75
- 3060, 3070, 3080, 3090: compute_86,sm_86

Under **“Linker”** property, change the **“Additional Library Directories”** to include the right CUB path.

After that, set the following System Wide Variable if you don’t have it yet:

**Variable name:** NVTOOLSEXT_PATH
**Value:** C:\Program Files\NVIDIA Corporation\NvToolsExt\

Finally, the program must run on the **“release” channel, on x64**. 

## Linux Guide (Ubuntu)

For Linux, just download the source code and edit the CMakeLists.txt file. The important lines that must be edited are the following ones:

- INCLUDE_DIRECTORIES("/media/Data/CUDA_SIMD_CMAKE/CUDA_ImCod/cub" "/media/Data/CUDA_SIMD_CMAKE/CUDA_ImCod/lib" "/usr/local/cuda/include" "/usr/local/cuda/lib64")

- LINK_DIRECTORIES("/media/Data/CUDA_SIMD_CMAKE/CUDA_ImCod/cub" "/media/Data/CUDA_SIMD_CMAKE/CUDA_ImCod/lib" "/usr/local/cuda/include" "/usr/local/cuda/lib64")

Changing the absolute paths to those that correspond depending on where you are storing the source code.

- set (CMAKE_CUDA_FLAGS -ccbin\ "/usr/bin/gcc"\ -gencode=arch=compute_75,code=\\"sm_75,compute_75\\"\ -rdc=true\ -maxrregcount=0\ --machine\ 64\ -cudart\ static\ -use_fast_math\ -Xptxas\ -v\ -lineinfo )

This is important: change the arch 75 code to the one that corresponds depending on your Nvidia GPU:

- 1050, 1060, 1070, 1080: compute_61,sm_61
- 2060, 2070, 2080: compute_75,sm_75
- 3060, 3070, 3080, 3090: compute_86,sm_86

After that, run the following commands to compile and create the binary file:
- cmake.
- cmake --build .

This will generate a binary called PICSONG, which is the executable of this codec.

## Codec instructions and examples

To execute the codec, you can first access the different parameters it offers. If you execute the application with parameter “-h”, you will get the set of instructions available for it, which are:

- **h:** Shows up this information
- **wl:** Sets the amount of Wavelet transforms that are used in the codification. Only available for CODING.
- **cp [2..3]:** Sets the amount of coding passes performed by the algorithm (either 2 or 3). 3 is deprecated.
- **type [0..1]:** If set to 0, the algorithm will make a Lossless DWT 5/3 transform. If set to 1, DWT 9/7 will be used, and quantization steps will be taken in consideration.
- **qs [0..1]:** This value will be used to make a quantization over the DWT coefficients after the transformed is applied. Please take into account that this value will only be used when the Lossy DWT is selected.
- **i:** This value is used to input the file to be coded/decoded. Take into account that currently only PGM, 1 band, greyscale images with .PGM "P5" format are compatible with the codec.
- **o:** This value is used to input the output file (either the coded file or the decoded one). The output is RAW always, with the addition of a ppm header for images.
- **cbWidth:** Sets the DWT tile width sizing. Value recommended: 64.
- **cbLength:** Sets the DWT tile length sizing. Value recommended: 18
- **cd [0..1]:** 0 for Coding, 1 for Decoding.
- **xSize:** Length of the image in the x-axis.
- **ySize:** Lenght of the image in the y-axis.
- **video[0..1]:** Tells the coder whether we are dealing with an image or a RAW video/multispectral image.
- **frames:** Tells the coder how many frames the video has (or components from a multispectral image).
- **LUTFolder:** Sets where the folder with the LUT files is located.
- **isRGB[0..1]:** Tells whether the image is RGB or not.
- **numberOfStreams:** Sets the amount processing lanes used in the codec. This parameter is limited by the GPU Architecture.
- **k:** Positive number that controls the complexity ratio in terms of throughput. Maximum value: 65.535

The examples provided on the Resources folder are a RGB image (2560x2048) and a 4K (4096x2048) RGB video (that must be downloaded separately from this link: https://drive.google.com/file/d/1lWUQi0kFN3Dn8RT69p5miIdsQQMcd12G/view?usp=sharing ). Please remember that **this codec only admits RAW files in Planar mode** (i.e., if a video or image is in RGB, the channels are placed in the following order: first all the red pixels, then all the green pixels and lastly all the blue pixels). If you choose to code an image with the color channels mixed, the compression quality will be poor and artifacts are bound to appear. It’s important to note that the information to be provided to the codec must have certain proportions. The Discrete Wavelet Transform requires the sizes, both x and y, to be divisible by 2 as many times as the user selects in the -wl parameter. For a decomposition level of 5, which is the usual decomposition we used throughout the development, for both sizes it suits the needs of the WL:

- 4096 – 2048 – 1024 – 512 – 254
- 2048 – 1024 – 512 – 256 – 128
- 2560 – 1280 – 640 – 320 – 160

Hence, these files are appropriate for the codec. As LUT files, which are side information files containing statistical data for the BPC-PaCo engine, we have included versions for both, lossy and lossless, and for different values of the complexity scalability engine, depending on how many bitplanes are coded in block, controlled by the parameter -k. Please note that a -k of 5 does not mean that 5 bitplanes are coded in block for each codeblock. Instead, parameter -k controls a variable of the CS equation explained in our latest article:  https://www.sciencedirect.com/science/article/pii/S0923596521002459

If you wish to test the codec, please use the following instructions as general examples to code/decode in lossy/lossless mode each of the files. Examples are provided for the Linux version.

### Image – Lossy: 
- ./PICSONG -wl 5 -cp 2 -type 1 -qs 1 -i "Resources/n1_RGB.3_2560_2048_1_0_8_0_0_1.raw" -o "Resources/n1_RGB.3_2560_2048_1_0_8_0_0_1.enc" -cbWidth 64 -cbHeight 18 -cd 0 -xSize 2048 -ySize 2560 -video 0 -isRGB 1 -LUTFolder LUT/n1_lossy/ -k 0
- ./PICSONG -i "Resources/n1_RGB.3_2560_2048_1_0_8_0_0_1.enc" -o "Resources/n1_RGB_DECODED.3_2560_2048_1_0_8_0_0_1.raw" -cd 1 -video 0 -LUTFolder LUT/n1_lossy/
### Image – Lossless:
- ./PICSONG -wl 5 -cp 2 -type 0 -qs 1 -i "Resources/n1_RGB.3_2560_2048_1_0_8_0_0_1.raw" -o "Resources/n1_RGB.3_2560_2048_1_0_8_0_0_1.enc" -cbWidth 64 -cbHeight 18 -cd 0 -xSize 2048 -ySize 2560 -video 0 -isRGB 1 -LUTFolder LUT/n1_lossless/ -k 0
- ./PICSONG -i "Resources/n1_RGB.3_2560_2048_1_0_8_0_0_1.enc" -o "Resources/n1_RGB_DECODED.3_2560_2048_1_0_8_0_0_1.raw" -cd 1 -video 0 -LUTFolder LUT/n1_lossless/
### Video – Lossy:
- ./PICSONG -wl 5 -cp 2 -type 1 -qs 1 -i "Resources/video.raw" -o "Resources/video.enc" -cbWidth 64 -cbHeight 18 -cd 0 -xSize 4096 -ySize 2048 -video 1 -frames 2090 -numberOfStreams 20 -isRGB 1 -LUTFolder LUT/video_lossy/ -k 0
- ./PICSONG -i "Resources/video.enc" -o "Resources/video_DEC.enc" cd 1 -video 1 -numberOfStreams 20 -LUTFolder LUT/video_lossy/ -k 0
### Video – Lossless:
- ./PICSONG -wl 5 -cp 2 -type 0 -qs 1 -i "Resources/video.raw" -o "Resources/video.enc" -cbWidth 64 -cbHeight 18 -cd 0 -xSize 4096 -ySize 2048 -video 1 -frames 2090 -numberOfStreams 20 -isRGB 1 -LUTFolder LUT/video_lossless/ -k 0
- ./PICSONG -i "Resources/video.enc" -o "Resources/video_DEC.enc" cd 1 -video 1 -numberOfStreams 20 -LUTFolder LUT/video_lossless/ -k 0

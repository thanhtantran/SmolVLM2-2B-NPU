## SmolVLM2-2.2B VLM for RK3588 NPU (Orange Pi 5 Plus 4GB). <br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) <br/>
Paper: https://huggingface.co/blog/smolvlm2 <br/>
Hugging face: https://huggingface.co/blog/smolvlm2

------------

## Introduction

LLMs (Large Language Models) are neural networks trained on extensive text datasets to comprehend and produce language.<br>
VLMs (Vision-Language Models) incorporate a visual encoder, allowing the model to process images and text simultaneously.<br> 
A combined VLM+LLM system is often referred to as a multimodal model.

These models can be large—hundreds of millions to billions of parameters—which impacts accuracy, memory use, and runtime speed.<br>
On edge devices like the RK3588, available RAM and compute are limited, and even the NPU has strict constraints on supported operations.<br>
Because of this, models typically need to be quantised or simplified to fit.

Performance is usually expressed in tokens (words) per second.<br>
Once converted to RKNN, parts of the model can run on the NPU, improving speed.<br>
Despite these limits, models like SmolVLM2-2.B run well on the RK3588 because the NPU efficiently accelerates the heavy math, and the vision encoder can be optimised. This makes advanced multimodal AI feasible on small, power-efficient devices.

------------

## Model performance benchmark (FPS)

All models, with C++ examples, can be found on the Q-engineering GitHub.<br><br>
All LLM models are quantized to **w8a8**, while the VLM vision encoders use **fp16**.<br>

| model         | RAM (GB)<sup>1</sup> | llm cold sec<sup>2</sup> | llm warm sec<sup>3</sup> | vlm cold sec<sup>2</sup> | vlm warm sec<sup>3</sup> | Resolution | Tokens/s |
| --------------| :--: | :-----: | :-----: | :--------: | :-----: | :--------:  | :--------: |
| [Qwen3-2B](https://github.com/Qengineering/Qwen3-VL-2B-NPU) | 3.1 | 21.9 | 2.6 | 10.0  | 0.9 | 448 x 448 | 11.5 |
| [Qwen3-4B](https://github.com/Qengineering/Qwen3-VL-4B-NPU) | 8.7 | 49.6 | 5.6 | 10.6  | 1.1 | 448 x 448 | 5.7 |
| [Qwen2.5-3B](https://github.com/Qengineering/Qwen2.5-VL-3B-NPU) | 4.8 | 48.3 |  4.0 | 17.9  | 1.8 | 392 x 392 | 7.0 |
| [Qwen2-7B](https://github.com/Qengineering/Qwen2-VL-7B-NPU) | 8.7 | 86.6 |   34.5 | 37.1  | 20.7 | 392 x 392 | 3.7 |
| [Qwen2-2.2B](https://github.com/Qengineering/Qwen2-VL-2B-NPU) | 3.3 | 29.1 |   2.5 | 17.1  | 1.7 | 392 x 392 | 12.5 |
| [InternVL3-1B](https://github.com/Qengineering/InternVL3-NPU) | 1.3 |  6.8 |   1.1 | 7.8    | 0.75 | 448 x 448 | 30 |
| [SmolVLM2-2.2B](https://github.com/Qengineering/SmolVLM2-2B-NPU) | 3.4 | 21.2 |   2.6 | 10.5   | 0.9  | 384 x 384 | 11 |
| [SmolVLM2-500M](https://github.com/Qengineering/SmolVLM2-500M-NPU) | 0.8 |  4.8 |   0.7 | 2.5    | 0.25 | 384 x 384 | 31 |
| [SmolVLM2-256M](https://github.com/Qengineering/SmolVLM2-256M-NPU) | 0.5 |  1.1 |   0.4 | 2.5    | 0.25 | 384 x 384 | 54 |

<sup>1</sup> The total used memory; LLM plus the VLM. <br>
<sup>2</sup> When an llm/vlm model is loaded for the first time from your disk to RAM or NPU, it is called a cold start.<br>
The duration depends on your OS, I/O transfer rate, and memory mapping.<br> 
<sup>3</sup> Subsequent loading (warm start) takes advantage of the already mapped data in RAM. Mostly, only a few pointers need to be restored.<br><br>

------------

## Instruction

### Installing the dependencies.
Start with the usual 
```bash
sudo apt-get update 
sudo apt-get upgrade
sudo apt-get install cmake wget curl build-essential
```

### Clone the repo.
```bash
git clone https://github.com/Qengineering/SmolVLM2-2B-NPU.git
cd SmolVLM2-2B-NPU
```

### Install OpenCV
```bash
sudo apt install -y python3-opencv libopencv-dev
```

##### Test it
```bash
orangepi@orangepi5plus:~/SmolVLM2-2B-NPU$ python3 -c "import cv2; print('OpenCV installed successfully'); print(cv2.__version__)"
OpenCV installed successfully
4.5.4
```


### Copy RKLLM, RKNN
To run SmolVLM2-2B, you need to have the **rkllm-runtime** library version **1.2.2** (or higher) installed, as well as the **rknpu driver** version **0.9.8**.<br>
If you don't have these on your machine, or if you have a lower version, you need to install them.<br>
We have provided the correct versions in the repo.<br>
```bash
sudo cp aarch64/library/*.so /usr/local/lib
sudo cp aarch64/include/*.h /usr/local/include
```
### Download the LLM and VLM model.
Download the two needed files (2.7 GB) from our Vietnodes.com server:<br>
[smolvlm2-2.2b_vision_fp16_rk3588.rknn](https://vietnodes.com/wl/?id=YR9v0XYxJF0NtQIb4BxA3zpEsTuoNOwM)
[smolvlm2-2.2b-instruct_w8a8_rk3588.rkllm](https://vietnodes.com/wl/?id=vDhnZui1LMVuBQ5fAxdxRgRuMGAgsphK)
Copy both into `/models` folder.


### Building the app.
```
mkdir build && cd build
cmake ..
make -j8
```
### Running the app.
The app has the following arguments.
```bash
VLM_NPU Picture RKNN_model RKLLM_model NewTokens ContextLength
```
| Argument   | Comment |
| --------------| --  |
| picture | The image. Provide a dummy if you don't want to use an image | 
| RKNN_model | The visual encoder model (vlm) | 
| RKLLM_model | The large language model (llm) | 
| NewTokens | This sets the maximum number of new tokens. Optional, default 2048| 
| ContextLength | This specifies the maximum total number of tokens the model can process. Optional, default 4096| 

<br>In the context of the Rockchip RK3588 LLM (Large Language Model) library, the parameters NewTokens and ContextLength both control different limits for text generation, and they're typical in LLM workflows.<br>
**NewTokens**<br> 
This sets the maximum number of tokens (pieces of text, typically sub-word units) that the model is allowed to generate in response to a prompt during a single inference round. For example, if set to 300, the model will not return more than 300 tokens as output, regardless of the prompt length. It's important for controlling generation length to avoid too-short or too-long responses, helping manage resource use and output size.<br>
**ContextLength**<br>
This specifies the maximum total number of tokens the model can process in one go, which includes both the prompt (input) tokens and all generated tokens. For example, if set to 2048 and your prompt already uses 500 tokens, the model can generate up to 2048-500 = 1548 new tokens. This is a hardware and architecture constraint set during model conversion and deployment, as the context window cannot exceed the model's design limit (for instance, 4096 or 8192 tokens depending on the model variant).

A typical command line can be:
```bash
VLM_NPU ./Moon.jpg ./models/smolvlm2-2.2b-instruct_w8a8_rk3588.rknn ./models/smolvlm2-2.2b_vision_fp16_rk3588.rkllm 2048 4096
```
The NewTokens (2048) and ContextLength (4096) are optional and can be omitted.

### Using the app.
If you want to talk about the picture, you need to include the token `<image>` in your prompt once.<br>
The app remembers the dialogue until you give the token `<clear>`.<br>
With `<exit>`, you leave the application.


## VLM examples.
![Alt text](https://github.com/user-attachments/assets/6d297a34-c516-4cb1-be4a-bca471d40fa6)
<br><br>**User**:\<image\>Describe the image.<br><br>
**Answer**: In this captivating digital painting, an astronaut is comfortably seated on the surface of the moon, with their feet propped up on a green cooler. The astronaut, clad in a pristine white spacesuit, has a relaxed posture, suggesting they are enjoying a leisurely moment away from the hustle and bustle of space exploration.

The backdrop to this tranquil scene is a breathtaking view of Earth, which is visible in the distance. The planet's blue oceans contrast beautifully with its brown landmasses, while its white clouds add depth to the atmosphere. The moon itself forms a stark, cratered landscape that serves as an excellent contrast to the vibrant colors of Earth.

The astronaut's gaze is directed towards the Earth, indicating their fascination with our home planet. Their relaxed demeanor and the presence of the cooler suggest they might be taking a break from their spacewalk or perhaps enjoying a well-deserved moment on the lunar surface. The painting beautifully captures this serene moment in space exploration, highlighting both the astronaut's peaceful solitude and the awe-inspiring beauty of Earth.

------------
## Credits

- Original code: https://github.com/Qengineering/SmolVLM2-2B-NPU
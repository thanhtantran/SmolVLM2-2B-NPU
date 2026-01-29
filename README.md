[🇺🇸 English version](README-en.md)

## SmolVLM2-2.2B VLM cho RK3588 NPU (Orange Pi 5 Plus 4GB)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  
Bài báo: https://huggingface.co/blog/smolvlm2  
Hugging Face: https://huggingface.co/blog/smolvlm2

------------

## Giới thiệu

LLM (Large Language Models – Mô hình ngôn ngữ lớn) là các mạng nơ-ron được huấn luyện trên tập dữ liệu văn bản khổng lồ nhằm hiểu và sinh ngôn ngữ.
VLM (Vision-Language Models – Mô hình thị giác–ngôn ngữ) tích hợp thêm bộ mã hóa hình ảnh, cho phép mô hình xử lý đồng thời cả hình ảnh và văn bản.
Hệ thống kết hợp VLM + LLM thường được gọi là mô hình đa phương thức (multimodal model)

Các mô hình này có thể rất lớn — từ hàng trăm triệu đến hàng tỷ tham số — ảnh hưởng trực tiếp đến độ chính xác, mức sử dụng bộ nhớ và tốc độ chạy. Chúng ta sẽ thường thấy các mô hình này chạy trên các máy chủ lớn với card màn hình giá trị cao, nhiều khi đầu tư cả hệ thống lên tới hàng chục triệu thậm chí hàng trăm triệu. Trên các thiết bị edge như Orange Pi sử dụng RK3588, tài nguyên RAM và khả năng tính toán bị giới hạn, và ngay cả NPU cũng có các ràng buộc nghiêm ngặt về các phép toán được hỗ trợ. Vì vậy, mô hình thường cần được lượng tử hóa hoặc tinh giản để có thể chạy được.

Mô hình SmolVLM2 này là một ví dụ của việc lượng tử hóa hoặc tinh giản tối đa để chạy trên các thiết bị edge device. Sau khi chuyển đổi sang định dạng RKNN, một phần mô hình có thể chạy trên NPU, giúp tăng tốc đáng kể. Sau khi loại bỏ các giới hạn này, những mô hình như SmolVLM2-2.2B vẫn chạy tốt trên RK3588 nhờ NPU tăng tốc hiệu quả các phép toán nặng và bộ mã hóa thị giác có thể được tối ưu. Điều này giúp AI đa phương thức tiên tiến có thể triển khai trên các thiết bị nhỏ gọn, tiết kiệm điện năng.

------------

## Bảng benchmark hiệu năng (FPS)

Tất cả các mô hình kèm ví dụ C++ đều có trên GitHub của Q-engineering.  

Tất cả các mô hình LLM đều được lượng tử hóa **w8a8**, trong khi bộ mã hóa thị giác của VLM sử dụng **fp16**.

| Model | RAM (GB) | LLM cold (giây) | LLM warm (giây) | VLM cold (giây) | VLM warm (giây) | Độ phân giải | Token/giây |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Qwen3-2B | 3.1 | 21.9 | 2.6 | 10.0 | 0.9 | 448x448 | 11.5 |
| Qwen3-4B | 8.7 | 49.6 | 5.6 | 10.6 | 1.1 | 448x448 | 5.7 |
| Qwen2.5-3B | 4.8 | 48.3 | 4.0 | 17.9 | 1.8 | 392x392 | 7.0 |
| Qwen2-7B | 8.7 | 86.6 | 34.5 | 37.1 | 20.7 | 392x392 | 3.7 |
| Qwen2-2.2B | 3.3 | 29.1 | 2.5 | 17.1 | 1.7 | 392x392 | 12.5 |
| InternVL3-1B | 1.3 | 6.8 | 1.1 | 7.8 | 0.75 | 448x448 | 30 |
| SmolVLM2-2.2B | 3.4 | 21.2 | 2.6 | 10.5 | 0.9 | 384x384 | 11 |
| SmolVLM2-500M | 0.8 | 4.8 | 0.7 | 2.5 | 0.25 | 384x384 | 31 |
| SmolVLM2-256M | 0.5 | 1.1 | 0.4 | 2.5 | 0.25 | 384x384 | 54 |

------------

## Hướng dẫn cài đặt

### Cài đặt các thư viện phụ thuộc
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install cmake wget curl build-essential
```

### Clone repository
```bash
git clone https://github.com/Qengineering/SmolVLM2-2B-NPU.git
cd SmolVLM2-2B-NPU
```

### Cài đặt OpenCV
```bash
sudo apt install -y python3-opencv libopencv-dev
```

### Kiểm tra OpenCV
```bash
python3 -c "import cv2; print('OpenCV installed successfully'); print(cv2.__version__)"
```

### Cài đặt RKLLM và RKNN
Để chạy SmolVLM2-2B, bạn cần **rkllm-runtime >= 1.2.2** và **rknpu driver >= 0.9.6**.  
Các phiên bản phù hợp đã được cung cấp sẵn trong repo.

```bash
sudo cp aarch64/library/*.so /usr/local/lib
sudo cp aarch64/include/*.h /usr/local/include
```

### Tải model LLM và VLM
Tải 2 file model (~1.5GB) từ Vietnodes.com<br>
[smolvlm2-2.2b_vision_fp16_rk3588.rknn](https://vietnodes.com/wl/?id=YR9v0XYxJF0NtQIb4BxA3zpEsTuoNOwM)<br>
[smolvlm2-2.2b-instruct_w8a8_rk3588.rkllm](https://vietnodes.com/wl/?id=vDhnZui1LMVuBQ5fAxdxRgRuMGAgsphK)<br>
 và chép vào thư mục `./model`.

### Build ứng dụng
```bash
mkdir build && cd build
cmake ..
make -j8
```

### Chạy ứng dụng
Cú pháp:
```bash
./VLM_NPU Picture RKNN_model RKLLM_model NewTokens ContextLength
```

**NewTokens**: số token tối đa sinh ra.  
**ContextLength**: tổng số token tối đa (prompt + output).

Ví dụ:
```bash
./VLM_NPU ./Moon.jpg ./models/smolvlm2-2.2b_vision_fp16_rk3588.rknn ./models/smolvlm2-2.2b-instruct_w8a8_rk3588.rkllm 2048 4096
```

### Sử dụng
- Dùng `<image>` trong prompt để nói về hình ảnh  
- `<clear>` để xóa hội thoại  
- `<exit>` để thoát chương trình  

------------

## Ghi công
- Mã nguồn gốc: https://github.com/Qengineering/SmolVLM2-2B-NPU

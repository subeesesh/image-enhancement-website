# AI Image Enhancement Website

This web application uses a pre-trained ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) model to enhance images with 4x super-resolution.

## Features

- Upload images in JPG, JPEG, or PNG format
- 4x Super-resolution enhancement
- Download enhanced images
- User-friendly web interface

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- Flask
- NumPy

## Directory Structure

```
├── app.py                  # Flask web application
├── RRDBNet_arch.py         # Model architecture definition
├── models/
│   └── RRDB_ESRGAN_x4.pth  # Pre-trained model
├── templates/
│   ├── index.html          # Home page template
│   └── result.html         # Results page template
├── uploads/                # Temporary storage for uploaded images
└── results/                # Temporary storage for enhanced images
```

## Installation

1. Clone this repository:
   ```
   git clone <your-repository-url>
   cd <repository-name>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have your pre-trained model file (`RRDB_ESRGAN_x4.pth`) in the `models/` directory.

## Usage

1. Start the web server:
   ```
   python app.py
   ```

2. Open your web browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

3. Upload an image through the web interface and click "Enhance Image".

4. After processing, you'll be redirected to the results page where you can:
   - View a comparison between original and enhanced images
   - Download the enhanced image
   - Choose to enhance another image

## Requirements File

Create a `requirements.txt` file with the following dependencies:

```
flask==2.0.1
numpy==1.21.0
opencv-python==4.5.3.56
torch==1.9.0
torchvision==0.10.0
```

## Notes

- Processing large images may take time depending on your hardware.
- The application is set to use CUDA if available, otherwise it will fall back to CPU.
- Maximum upload file size is limited to 16MB by default.
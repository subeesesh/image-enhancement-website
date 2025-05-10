import os
import uuid
import torch
import numpy as np
import cv2
from flask import Flask, request, render_template, send_from_directory, url_for, flash, redirect
from werkzeug.utils import secure_filename
import RRDBNet_arch as arch

# Make sure we're looking for templates in the right place
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = "super_secret_key_for_image_enhancement"

# Create necessary directories
app.config['UPLOAD_FOLDER'] = os.path.join(static_dir, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(static_dir, 'results')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB

# Ensure directories exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Create the upload and results folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Check if CUDA is available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the pre-trained model
model_path = 'models/RRDB_ESRGAN_x4.pth'
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)
print(f"Model loaded from: {model_path}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Save the file with a unique filename
            original_filename = secure_filename(file.filename)
            base_filename = os.path.splitext(original_filename)[0]
            unique_id = str(uuid.uuid4())[:8]
            unique_filename = f"{base_filename}_{unique_id}"
            
            # Save the original file
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename + '.jpg')
            file.save(input_path)
            
            # Verify the file was saved correctly
            if not os.path.exists(input_path):
                flash(f'Error saving uploaded file to {input_path}')
                return redirect(url_for('index'))
                
            print(f"Image saved to: {input_path}")
            
            # Read and preprocess the image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                flash('Error reading uploaded image. The file might be corrupted.')
                return redirect(url_for('index'))
                
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)
            
            print("Image preprocessed and loaded to device successfully")
            
            # Run inference
            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()
            
            # Save the result
            output_filename = unique_filename + '_enhanced.png'
            output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
            success = cv2.imwrite(output_path, output)
            if not success:
                flash('Error saving enhanced image')
                return redirect(url_for('index'))
                
            print(f"Enhanced image saved to: {output_path}")
            
            # Generate static URLs for images
            original_filename_only = os.path.basename(input_path)
            original_static_url = f"/static/uploads/{original_filename_only}"
            enhanced_static_url = f"/static/results/{output_filename}"
            
            # Return the result page - try the simple template first
            try:
                # Try the simple result template first
                return render_template('simple_result.html', 
                                    original=original_filename_only,
                                    enhanced=output_filename,
                                    original_url=original_static_url,
                                    enhanced_url=enhanced_static_url)
            except Exception as simple_template_error:
                print(f"Error with simple template: {str(simple_template_error)}")
                try:
                    # Fallback to the original template
                    return render_template('result.html', 
                                        original=original_filename_only,
                                        enhanced=output_filename,
                                        original_url=original_static_url,
                                        enhanced_url=enhanced_static_url)
                except Exception as template_error:
                    flash(f'Error rendering templates: {str(template_error)}')
                    return redirect(url_for('index'))
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details: {error_details}")
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG file.')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(os.path.abspath(app.config['RESULTS_FOLDER']), filename)

@app.route('/direct-result/<filename>')
def direct_result(filename):
    """Direct route to display a specific enhanced image"""
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_from_directory(os.path.abspath(app.config['RESULTS_FOLDER']), filename)
    else:
        return f"File not found: {file_path}", 404

@app.route('/show-results')
def show_results():
    """Display all enhanced images"""
    results_path = os.path.abspath(app.config['RESULTS_FOLDER'])
    if not os.path.exists(results_path):
        return "Results directory not found", 404
        
    files = os.listdir(results_path)
    enhanced_images = [f for f in files if f.endswith('.png')]
    
    html = "<h1>Enhanced Images</h1>"
    html += "<ul>"
    for img in enhanced_images:
        img_url = url_for('result_file', filename=img)
        html += f'<li><img src="{img_url}" width="300"><br><a href="{img_url}" download>Download {img}</a></li>'
    html += "</ul>"
    
    return html

if __name__ == '__main__':
    app.run(debug=True)
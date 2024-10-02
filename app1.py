from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
import cripser as cr
import tcripser as tcr
from persim.persistent_entropy import *

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def compute_entropies(persistence_diagram, epsilon):
    entropy_vals = []
    for i in range(3):
        entropy_vals.append(persistent_entropy(persistence_diagram, hom_dim=i, epsilon=epsilon))
    return entropy_vals

def classify_result(betti_numbers_v, betti_numbers_t):
    # Define thresholds for classification
    betti_thresholds = {
        'betti_0': 300,
        'betti_1': 1000,
        'betti_2': 100
    }

    # Classify the result based on Betti numbers
    if betti_numbers_v[2] > betti_thresholds['betti_2'] or betti_numbers_t[2] > betti_thresholds['betti_2']:
        return "High likelihood of Alzheimer's disease"
    elif betti_numbers_v[2] < betti_thresholds['betti_2'] and betti_numbers_t[2] < betti_thresholds['betti_2']:
        return "Cognitive normal "
    else:
        return "Mild cognitive impairment"

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/generate_persistence_diagram', methods=['POST'])
def generate_persistence_diagram():
    img_files = request.files.getlist('images')

    if not img_files:
        return jsonify({'status': 'error', 'message': 'No image files uploaded'})

    img_stack = []
    for img_file in img_files:
        img = Image.open(BytesIO(img_file.read())).convert('L')
        img_stack.append(np.array(img))

    img_stack = np.dstack(img_stack)

    # Apply distance transform
    dt_img_stack = np.zeros_like(img_stack, dtype=np.float64)  
    for i in range(img_stack.shape[-1]):
        dt_img_stack[:,:,i] = distance_transform_edt(img_stack[:,:,i])
        inverted_img = ~img_stack[:,:,i].astype(bool)

        dt_inverted_img = distance_transform_edt(inverted_img).astype(np.float64)  
        dt_img_stack[:,:,i] -= dt_inverted_img  

    # Compute persistent homology for V-construction
    pd_v_construction = cr.computePH(dt_img_stack)

    # Compute persistent homology for T-construction
    pd_t_construction = tcr.computePH(dt_img_stack)

    # Calculate Betti numbers
    betti_numbers_v = [len(pd_v_construction[pd_v_construction[:,0] == i]) for i in range(3)]
    betti_numbers_t = [len(pd_t_construction[pd_t_construction[:,0] == i]) for i in range(3)]
# Calculate entropies
    epsilon = 0.2  # Adjust as needed
    entropies_v = compute_entropies(pd_v_construction, epsilon)
    entropies_t = compute_entropies(pd_t_construction, epsilon)

    # Classify the result
    result = classify_result(betti_numbers_v, betti_numbers_t)

    return jsonify({
        'status': 'success',
        'pd_v_construction': pd_v_construction.tolist(),
        'pd_t_construction': pd_t_construction.tolist(),
        'betti_numbers_v': betti_numbers_v,
        'betti_numbers_t': betti_numbers_t,
        'entropies_v': entropies_v,
        'entropies_t': entropies_t,
        'result': result
    })


if __name__ == '__main__':
    app.run(debug=True)

import numpy as np

input_file_path = "dinoSparseRing/dinoSR_par.txt"

f = open(input_file_path, 'r')
num_images = int(f.readline())
data = f.read()

lines = data.split('\n')
data_array_str = []

for i in range(num_images):
    line = lines[i]
    # first word is image path, not needed right now, so start at index 1
    data_array_str.append(line.split(' ')[1:])

data_array = [list(map(float, data_array_str[i])) for i in range(len(data_array_str))]

data = np.array(data_array)
# 16 images, 
# 9 parameters for calibration, 9 parameters for rotation, 3 parameters for translation
# total of 21 parameters
# FOR DINO SPARSE SET
assert(data.shape == (16, 21))

# CHECK IF THIS IS THE RIGHT FLATTENING
center_calibration_inv = np.linalg.inv(data[0,:9].reshape(3,3))

# CHECK IF THIS IS THE RIGHT FLATTENING
center_rotation_inv = np.linalg.inv(data[0,9:18].reshape(3,3))

center_t = data[0,18:]
centered_calibration_matrices = np.zeros((num_images, 3, 3))
centered_rotation_matrices = np.zeros((num_images, 3, 3))
centered_translation_vectors = np.zeros((num_images, 3))
for i in range(num_images):
    centered_calibration_matrices[i] = center_calibration_inv * (data[i,:9]).reshape(3,3)
    centered_rotation_matrices[i] = center_rotation_inv * (data[i,9:18]).reshape(3,3)
    centered_translation_vectors[i] = data[i,18:] - center_t

print(centered_calibration_matrices[0])
assert(np.all(centered_calibration_matrices[0] == np.identity(3)))
assert(np.all(centered_rotation_matrices[0] == np.identity(3)))
assert(np.all(centered_translation_vectors[0] == np.zeros((1,3))))



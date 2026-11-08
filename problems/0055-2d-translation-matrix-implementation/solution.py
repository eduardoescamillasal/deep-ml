import numpy as np
def translate_object(points, tx, ty):
	translated_points = []
	points = np.asarray(points)
	translation_mat = np.eye(3)
	translation_mat[0,2] = tx
	translation_mat[1,2] = ty
	for point in points:
		homogeneous_pnt = np.array([point[0], point[1], 1.0])
		translated = translation_mat @ homogeneous_pnt
		translated_points.append(translated[:2].tolist())
	return translated_points

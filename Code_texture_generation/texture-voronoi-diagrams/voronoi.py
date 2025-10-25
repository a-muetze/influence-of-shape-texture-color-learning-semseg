
""""
General setup for voronoi diagram
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.voronoi_plot_2d.html
Generate polygones from voronoi cells
https://gist.github.com/pv/8036995

Draw and fill voronoi diagrams
"""
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import cv2

import os
from PIL import Image, ImageDraw
from tqdm import tqdm



def pathfind(i, root_prefix, split): 
    path = []
    texture = f'{root_prefix}_{i}/{split}'
    for (root, dirs, files) in os.walk(texture):
        for texture_file in files:
            s = texture_file.split('_')
            if s[0] == f'final{i}':
                path.append(os.path.join(root, texture_file))
            elif s[2] == 'patched' and s[3] == str(i):
                path.append(os.path.join(root, texture_file))
    # If no complete image was generated. Remove afterwards
    if not path:
        exit(f"No texture images found at {texture}")
    if len(path)-1 == 0:
        u = 0
    else:
        u = rng.integers(0, len(path)-1)
    print(f"label_id: {i}, texture_img: {u}")
    return path[u], i




def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Limit infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices),


def randomize_voronoi_diagram(num_points, dim, name, root, split, num_ids=19, label_list = [], result_dir = "results"):
    '''
    Generate voronoi centers and fill each cell with a texture image of a certain class

    Parameters
    ----------
    num_points: Number of voronoi centers
    dim: dimension of the diagram (e.g. 2D)
    name: name to identify the generated diagram

    Returns
    ---------
    Voronoi Diagram
    '''
    fig, ax1 = plt.subplots(dpi=300)
    points = rng.random((num_points, dim))*(2048, 1024)

    # add 4 distant dummy points
    points = np.append(points, [[2440, 1440], [-0, 1440], [2440, -0], [-0, -0]], axis=0)
    vor = Voronoi(points)

    voronoi_plot_2d(vor, ax1, show_vertices=False)
    plt.xlim([0, 2048]), plt.ylim([0, 1024])

    regions, vertices = voronoi_finite_polygons_2d(vor)

    print(f'region {len(regions)}')
    voronoi_img = Image.new('RGB', (2048, 1024), 0)
    voronoi_label = Image.new('L', (2048, 1024), 0)
    if label_list:
        rng.shuffle(label_list)
    else:
        label_list = [*range(num_ids)]
        rng.shuffle(label_list)
    count = 0
    for region in regions:
        if region:  # valid region (not empty polygonpoint list) and not -1 in region:
            polygon = vertices[region]

            label_id = label_list[count]
            count += 1
            if count >= len(label_list):
                rng.shuffle(label_list)
                count = 0

            texture_img_path, label_id = pathfind( label_id, root, split)
            voronoi_img, voronoi_label = cropping(voronoi_img,
                                                  voronoi_label,
                                                  texture_img_path,
                                                  label_id,
                                                  polygon)


    result_label_dir = os.path.join(result_dir + "_label", split)
    result_dir = os.path.join(result_dir, split)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_label_dir, exist_ok=True)
    voronoi_img.save(os.path.join(result_dir, str(name)) + ".png")
    voronoi_label.save(os.path.join(result_label_dir, str(name)) + "_train_id.png")
    # plt.gca().invert_yaxis()
    # ax1.plot(points[:-4, 0], points[:-4, 1], 'ro', markersize=2, zorder=10)

    # fig.savefig(f"Images/voronoi_structure_{name}.png")
    plt.close()


def cropping(voronoi, voronoi_label, path, label_id, polygon):
    '''
    Fill a single Voronoi cell with texture
    '''

    texture_img = Image.open(path)
    label_img = Image.new('L', (2048, 1024), int(label_id))

    polygon_mask = Image.new('L', (2048, 1024), 0)
    polygon_path =[tuple(np.int32(l)) for l in polygon]
    ImageDraw.Draw(polygon_mask).polygon(polygon_path, fill=(255), outline=(255))

    #fill voronoi diagram
    voronoi.paste(texture_img, (0,0), polygon_mask)
    voronoi_label.paste(label_img, (0,0), polygon_mask)

    return voronoi, voronoi_label




if __name__ == "__main__":
    # INFO: scripts expect a certain naming convention as implemented in polygon_semseg_contourfill.py
    # Adaption of the function "pathfind" is needed if contour filling is skipped in the procedure 
    cell_number = 100
    dimensionen = 2
    diagramm_amount = 2975
    root_path = "/home/textureImages/"
    base_prefix = "2023_06_05_upsampled_texture_images"
    split = 'val'
    result_dir = f"Voronoi_{base_prefix}"
    label_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    rng = np.random.default_rng(4224)  # random number generator for random handling in numpy

    for diagramm_id in tqdm(range(diagramm_amount)):
        randomize_voronoi_diagram(cell_number,
                                  dimensionen,
                                  diagramm_id,
                                  os.path.join(root_path, base_prefix),
                                  split,
                                  num_ids=11,
                                  label_list=label_list,
                                  result_dir=result_dir
                                  )
# end main


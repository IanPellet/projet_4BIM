# Images
import os
import PIL 
import numpy as np
import openslide

# Annotations
from xml.dom import minidom
import skimage.draw


def load_img(in_dir):
    '''
    Load images and transform it to np.arrays to be inputed in the model
    Parameters :
        in_dir -> directory containing the image files (str)
    Returned value :
        img_as_array -> list of images in in_dir represented as np.array of int (list of np.arrays)
    ''' 
    
    # list of image paths
    img_paths = sorted(
        [
            os.path.join(in_dir, fname) # join directory name with file name
            for fname in os.listdir(in_dir)
            if fname.endswith(".ndpi") # for ndpi files of the directory
        ]
    )
    
    # path -> OpenSlide image
    OpSl_img = [] # list of OpenSlide images
    for path in img_paths:
        temp_OpSl_img = openslide.OpenSlide(path) # open whole-slide image
        OpSl_img.append(temp_OpSl_img)
    
    # OpenSlide image -> PIL image
    # as we can't load full images we take an arbitrary level to extract
    PIL_img = []
    location = (0,0)
    level = 5
    for img in OpSl_img:
        size = img.level_dimensions[level]
        temp_PIL_img = img.read_region(location, level, size)
        #temp_PIL_img.show()
        img.close() # close image
        PIL_img.append(temp_PIL_img)
        
    # PIL image -> np.array
    img_as_array = [] # returned value, list of np.arrays
    for Pimg in PIL_img:
        temp_array = np.array(Pimg)
        img_as_array.append(temp_array)
        
    return img_as_array



def xml_to_vertices(xml, region_type = '0'):
    '''
    Returns the list of coordinates of the vertices of each regions 
    Parameters :
        xml -> parsed xml file
        region_type -> extract only positive '0' or negative regions '1'(str)
    Returned value :
        V_coord -> list of coordinates (dict) of all vertices in each region (2D array of dict)
    '''
    
    # extract positive regions
    regions = []
    for r in xml.getElementsByTagName("Region"):
        if r.getAttribute("NegativeROA")==region_type : regions.append(r)
            
    # extract vertices and there coordinates
    V_coord = [] # list of coord of each vertex in each region
    for r in regions :
        
        temp_vertices = r.getElementsByTagName("V")

        temp_coord = []
        for v in temp_vertices :
            coord = {"X": v.getAttribute('X'), "Y": v.getAttribute('Y')}
            temp_coord.append(coord)
            
        V_coord.append(temp_coord)
        
    return V_coord

def vertices_to_mask(img_shape, ds_rate, V_coord):
    mask = np.ones(img_shape) # creates an array of ones of the same shape of our image
    #ds_rate = 8 # downsample rate

    # list of polygons, each object is an array containing the coords of the px in a region of V_coord
    polygons = [] # 1 polygon / region

    for region in V_coord:
        x = [] # list of x coord in the region
        y = [] # list of y coord in the region

        # fill the x and y coord arrays
        for v in region:
            # as we dont take the full resolution slide of the image, 
            # we need to divide the coord of each vertice by the downsampling rate
            temp_x = int(v['X'])/ds_rate 
            temp_y = int(v['Y'])/ds_rate

            x.append(int(temp_x))
            y.append(int(temp_y))

        # sikimage computes the coord of each px in the region from the vertices defining 
        # the perimeter of the region
        polygons.append(skimage.draw.polygon(x,y)) 
        
    # modify mask so that each px in a region of V_coord is set to 0
    for p in range(len(polygons)):
        poly = np.transpose(polygons[p])

        for i in range(len(poly)):
            mask[(poly[i][0], poly[i][1])] = 0
        
    return mask

def load_annot(in_dir):
    '''
    Load annotation files and transform it to mask arrays to be inputed in the model
    Parameters :
        in_dir -> directory containing the annotation files (str)
    Returned value :
        mask_array -> list of images in in_dir represented as np.array of int (list of np.arrays)
    ''' 
    
    # list of annotations paths
    annot_paths = sorted(
        [
            os.path.join(in_dir, fname) # join directory name with file name
            for fname in os.listdir(in_dir)
            if fname.endswith(".annotations") # for annotations files of the directory
        ]
    )
    
    # path -> xml
    annot_xml = []
    for path in annot_paths:
        temp_xml = minidom.parse(path) # open xml
        annot_xml.append(temp_xml)
        
    # xml -> list of vertices
    annot_coords = []
    for xml in annot_xml:
        temp_coords = xml_to_vertices(xml, '0')
        annot_coords.append(temp_coords)
        
    # comment associer annot files w/ img ?
    # list of vertices -> mask array
    masks = []
    for coords in annot_coords:
        temp_mask = vertices_to_mask((1008, 840), 64, coords) # taille et ds_rate random
        masks.append(temp_mask)
    
    return masks
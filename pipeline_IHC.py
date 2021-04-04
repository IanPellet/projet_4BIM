"""@package pipeline_IHC
All methods needed to load data to be passed to the NN.
 
More details.
"""


# Images
import os
import PIL 
import numpy as np
import openslide

# Annotations
from xml.dom import minidom
import skimage.draw


def load_img(in_dir, one = False):
    """
    Load images and transform it to np.arrays to be inputed in the model

    :param in_dir: string, directory containing the image files.
    :param one: bool, True if :param in_dir: is the path to one image, default = False.
    
    :returns: dict of images in in_dir represented as np.array of int, indexed by image's name (str).
    """ 
    if one :
        img_paths = [in_dir]
    else:
        # list of image paths
        img_paths = sorted(
            [
                os.path.join(in_dir, fname) # join directory name with file name
                for fname in os.listdir(in_dir)
                if fname.endswith(".ndpi") # for ndpi files of the directory
            ]
        )
    
    # path -> OpenSlide image
    OpSl_img = {} # dict of OpenSlide images {name : OpSL image}
    for path in img_paths:
        temp_OpSl_img = openslide.OpenSlide(path) # open whole-slide image
        temp_img_name = path.split('/')[-1].split('.')[0] # retrives the image's name
        OpSl_img[temp_img_name] = temp_OpSl_img
    
    # OpenSlide image -> PIL image
    # as we can't load full images we take an arbitrary level to extract
    PIL_img = {}
    location = (0,0)
    level = 5
    for img_name in OpSl_img:
        img = OpSl_img[img_name]
        size = img.level_dimensions[level]
        temp_PIL_img = img.read_region(location, level, size)
        if one : temp_PIL_img.save(img_name+'.png')
        #temp_PIL_img.show()
        img.close() # close image
        PIL_img[img_name] = temp_PIL_img
        
    # PIL image -> np.array
    img_as_array = {} # returned value, list of np.arrays
    for img_name in PIL_img:
        Pimg = PIL_img[img_name]
        temp_array = np.array(Pimg)
        img_as_array[img_name] = temp_array
        
    return img_as_array



def xml_to_vertices(xml, region_type = '0'):
    """
    Returns the list of coordinates of the vertices of each regions 

    :param xml: parsed xml file.
    :param region_type: string, extract only positive '0' or negative regions '1'.
    
    :returns: list of coordinates (dict) of all vertices in each region (2D array of dict).
    """
    
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

def vertices_to_mask(img_shape, ds_rate, V_coord, png = False):
    """
    Creates a mask with the same shape as :param img_shape: representing the region of :param V_coord: with 0 and the rest of the array with 1.
    If :param png: is set to True, the created mask is an array of a RGB image with regions of :param V_coord: in white and the rest of the image in black.

    :param img_shape: tuple, shape of the image corresponding to :param V_coord:.
    :param ds_rate: int, down-sampling rate used to get an image of shape :param img_shape:.
    :param V_coord: list of coordinates (dict) of all vertices in each region (2D array of dict).
    :param pg: bool, True to generate a RGB image instead of a 0/1 array, default = False.
    
    :returns: int np.array, array representing the regions of :param V_coord:.
    """

    # create an array of ones of the same shape of our image
    if png : mask = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    else : mask = np.ones(img_shape) 
    	
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
    if png :
        for p in range(len(polygons)):
            poly = np.transpose(polygons[p])

            for i in range(len(poly)):
                #print((poly[i][1], poly[i][0]))
                mask[(poly[i][1], poly[i][0])] = [255, 255, 255]
    else :
        for p in range(len(polygons)):
            poly = np.transpose(polygons[p])

            for i in range(len(poly)):
                mask[(poly[i][0], poly[i][1])] = 0

    return mask

def load_annot(in_dir, img_dict):
    """
    Load annotation files and transform it to mask arrays to be inputed in the model
  
    :param in_dir: string, directory containing the annotation files.
    
    :returns: dict of images in in_dir represented as np.array of int, indexed by image's name (str).
    """ 
    
    # list of annotations paths
    annot_paths = sorted(
        [
            os.path.join(in_dir, fname) # join directory name with file name
            for fname in os.listdir(in_dir)
            if fname.endswith(".annotations") # for annotations files of the directory
            if fname.split('.')[0] in img_dict # load only the annotations corresponding to loaded img
        ]
    )
    #print("paths", annot_paths)
    
    # path -> xml
    annot_xml = {}
    for path in annot_paths:
        temp_xml = minidom.parse(path) # open xml
        temp_img_name = path.split('/')[-1].split('.')[0] # retrives the corresponding image's name
        annot_xml[temp_img_name] = temp_xml
    #print("xml", annot_xml)
        
    # xml -> list of vertices
    annot_coords = {}
    for img_name in annot_xml:
        xml = annot_xml[img_name]
        temp_coords = xml_to_vertices(xml, '0')
        annot_coords[img_name] = temp_coords
    #print("coords", annot_coords)
        
    # comment associer annot files w/ img ?
    # list of vertices -> mask array
    masks = {}
    for img_name in annot_coords:
        coords = annot_coords[img_name]
        img_shape = img_dict[img_name].shape
        temp_mask = vertices_to_mask(img_shape, 64, coords) # taille et ds_rate random
        masks[img_name] = temp_mask
    
    return masks
    
    
def png_mask(img_path, disp = False): 
    """
    Visualise the annotation corresponding to an image as a black and white png mask file.
    Saves the annotation and the image to png files.

    :param img_path: string, path of the ndpi image.
    :param disp: bool, True if you want to open the mask, default = False.

    :returns: PIL image, the annotation as a black and white RGB image.
    """

    image = load_img(img_path, True)
    annot_path = img_path.replace('ndpi', 'annotations')
    xml = minidom.parse(annot_path)
    coords = xml_to_vertices(xml, '0')
    name = img_path.split('/')[-1].split('.')[0]
    img_shape = image[name].shape
    mask_img = vertices_to_mask(img_shape, 32, coords, True)
    PILimg = PIL.Image.fromarray(mask_img, 'RGB')
    fname = name + '_mask.png'
    PILimg.save(fname)
    if disp:
        PILimg.show()
    return PILimg
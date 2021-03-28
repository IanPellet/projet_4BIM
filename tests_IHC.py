import unittest
import pipeline_IHC as pipe
from xml.dom import minidom


class Test_TrainingSet_improt(unittest.TestCase):

	def test_load_img(self):
		# load 2 images
		loaded = pipe.load_img("./test_data/data/")
		name = list(loaded.keys())[0]
		self.assertEqual(len(loaded), 2) # 2 images loaded
		self.assertGreater(len(loaded[name]), 1000) 
		self.assertGreater(len(loaded[name][0]), 1000)
		self.assertEqual(len(loaded[name][0][0]), 4) # RGBA images 

		# repository vide
		self.assertEqual(len(pipe.load_img("./test_data/nodata/")), 0)
		# repository sans ndpi
		self.assertEqual(len(pipe.load_img("./test_data/")), 0)
		# faute de frape dans le repository name
		with self.assertRaises(FileNotFoundError): 
			pipe.load_img("./test_data/datu/")

	def test_xml_to_vertices(self):
		# open XML file
		annot = minidom.parse("./test_data/data/test.annotations") 
		Vcoords = pipe.xml_to_vertices(annot)
		vert_dict = [[{'X': '20800', 'Y': '33300'}, {'X': '20700', 'Y': '33300'}, 
		{'X': '20700', 'Y': '33200'}, {'X': '20800', 'Y': '33200'}]]

		self.assertListEqual(Vcoords, vert_dict)
		self.assertListEqual(Vcoords[0], vert_dict[0])
		self.assertDictEqual(Vcoords[0][0], vert_dict[0][0])


if __name__ == '__main__':
	unittest.main()

import os
import sys
import unittest

from airrmap.application.config import AppConfig, SeqFileType
import airrmap.preprocessing.compute_selected_coords as compute_coords

class TestComputeSelectedCoords(unittest.TestCase):

    def setUp(self):
        self.env_name = 'test_env'
        self.base_folder = os.path.join(os.path.dirname(__file__), 'resource')
        self.app_cfg_fn = os.path.join(self.base_folder, 'test_app_config.yaml')
        self.app_cfg = AppConfig(self.app_cfg_fn)
        

    def test_get_coords(self):

        ## Init
        #seq1 = 'CARGYSSGWYYFDYW'
        #seq2 = 'CARDSSGWYYFDYW'
        seq1 = "{\"cdrh1\": {\"27\": \"G\", \"28\": \"Y\", \"29\": \"T\", \"30\": \"F\", \"35\": \"T\", \"36\": \"S\", \"37\": \"Y\", \"38\": \"G\"}, \"cdrh2\": {\"56\": \"I\", \"57\": \"S\", \"58\": \"A\", \"59\": \"Y\", \"62\": \"N\", \"63\": \"G\", \"64\": \"N\", \"65\": \"T\"}}" 
        seq2 = "{\"cdrh1\": {\"27\": \"Y\", \"28\": \"Y\", \"29\": \"T\", \"30\": \"F\", \"35\": \"T\", \"36\": \"S\", \"37\": \"Y\", \"38\": \"G\"}, \"cdrh2\": {\"56\": \"I\", \"57\": \"S\", \"58\": \"A\", \"59\": \"Y\", \"62\": \"N\", \"63\": \"G\", \"64\": \"N\", \"65\": \"T\"}}" 
        #                              ^ one residue different
        seq_list = [seq1, seq2]

        # Compute coordinates
        result_list = compute_coords.get_coords(
            env_name = self.env_name,
            seq_list=seq_list,
            convert_json=True,
            app_cfg=self.app_cfg
         )

        # Check response is a list with two items
        self.assertIsInstance(result_list, list, 'A list of results should be returned.')
        self.assertEqual(len(result_list), 2, 'All results should be returned.')

        # Check each item in the list
        for item in result_list:
            self.assertIsInstance(item['sys_coords_x'], float, "Result should contain 'sys_coords_x' (float)")
            self.assertIsInstance(item['sys_coords_y'], float, "Result should contain 'sys_coords_y' (float)")
            self.assertIsInstance(item['sys_coords_init_x'], float, "Result should contain 'sys_coords_init_x' (float)")
            self.assertIsInstance(item['sys_coords_init_y'], float, "Result should contain 'sys_coords_init_y' (float)")
            self.assertIsInstance(item['sys_coords_num_closest_anchors'], int, "Result should contain 'num_closest_anchors' (int)")
         

if __name__ == '__main__':
    unittest.main()
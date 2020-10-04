import unittest
import pochade
import os

class PochadeTest(unittest.TestCase):
    def test_palette_astronaut(self):
        path = os.path.join("data", "index.jpg")
        palette = pochade.palette(path)

        self.assertEqual(palette.shape[0], 6)pwd

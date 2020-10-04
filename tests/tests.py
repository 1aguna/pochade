import unittest
import pochade
import os

class PochadeTest(unittest.TestCase):
    def test_palette_astronaut(self):
        path = os.path.join("data", "index.jpg")
        palette = pochade.palette(path)

        self.assertEqual(palette.shape[0], 6)


[![1aguna](https://circleci.com/gh/1aguna/pochade.svg?style=svg)](https://app.circleci.com/pipelines/github/1aguna)
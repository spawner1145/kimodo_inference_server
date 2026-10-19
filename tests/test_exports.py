"""Validate BVH hierarchy/export for the three skeleton families without weights."""
import unittest
import torch

from kimodo.skeleton import build_skeleton
from kimodo.exports.bvh import motion_to_bvh


class ExportTests(unittest.TestCase):
    def test_all_skeleton_families(self):
        for count in (77,22,34):
            with self.subTest(joints=count):
                skel = build_skeleton(count)
                rots = torch.eye(3).expand(3,count,3,3).clone()
                roots = torch.zeros(3,3)
                roots[:,1] = 1
                text = motion_to_bvh(rots,roots,skeleton=skel,fps=30,standard_tpose=True)
                self.assertIn('Frames: 3',text)
                self.assertIn('Frame Time: 0.033333',text)
                self.assertEqual(text.count('JOINT '),count)


if __name__=='__main__':
    unittest.main()

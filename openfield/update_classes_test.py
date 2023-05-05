import unittest
from update_classes import File_operation as f
import io
import shutil


class TestUpdateClasses(unittest.TestCase):
    def setUp(self) -> None:
        self.musicfile = './nest_oopenfield_test_.music'
        shutil.copyfile('./nest_openfield_test.music', self.musicfile)

    def test(self):
        musicfile = "./nest_openfield_test.music"
        op = f("./sim_params.json", musicfile)

        op.update_port(1000)
        op.update_musicfile(musicfile)

        self.assertListEqual(
            list(io.open("./nest_openfield_test.music")),
            list(io.open("./nest_openfield_expected.music"))
        )


if __name__ == '__main__':
    unittest.main()

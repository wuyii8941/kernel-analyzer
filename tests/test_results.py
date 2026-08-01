import unittest

from scripts.check import main


class FinalResultsTest(unittest.TestCase):
    def test_package(self) -> None:
        main()


if __name__ == "__main__":
    unittest.main()

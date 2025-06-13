import unittest
from tests.test_hello_world import hello_world


class MainTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(hello_world(), "Hello, World!")


if __name__ == '__main__':
    unittest.main()

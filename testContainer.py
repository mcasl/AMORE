import unittest
from hypothesis import given, strategies as st
from container import Container


class TestContainer(unittest.TestCase):
    def test_Container_init_default_value(self):
        container = Container()
        self.assertTrue(container.data == [])

    @given(value=st.lists(st.integers()))
    def test_Container_init_non_default_value(self, value):
        container = Container(value)
        self.assertTrue(container.data == value)


if __name__ == '__main__':
    unittest.main()

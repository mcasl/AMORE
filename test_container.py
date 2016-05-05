import unittest

from hypothesis import given, strategies as st

from container import Container


class TestContainer(unittest.TestCase):
    """ Tests for Container class """

    def test_init_where_iterable_is_None(self):
        # Setup test
        container = Container()  # Test action
        self.assertEqual(len(container), 0)

    @given(input_data=st.lists(st.integers()))
    def test_init_where_iterable_is_a_list(self, input_data):
        # Setup test
        container = Container(input_data)
        # Test action
        self.assertTrue(container == input_data)

    @given(input_data=st.lists(st.integers()))
    def test_getitem_where_integer_item_returns_single_data_item(self, input_data):
        # Setup test
        container = Container(input_data)
        # Test action
        for data_index, data_item in enumerate(container):
            self.assertEqual(data_item, container.data[data_index])

    @given(input_data=st.lists(st.integers()))
    def test_getitem_where_slice_item_returns_Container_instance(self, input_data):
        # Setup test
        container = Container(input_data)
        # Test action
        self.assertTrue(isinstance(container[0:], type(container)))

    @given(input_data=st.lists(st.integers()))
    def test_getitem_where_slice_item_returns_container_data(self, input_data):
        # Setup test
        container = Container(input_data)
        # Test action
        self.assertEqual(container[0:], container.data)

    @given(input_data=st.lists(st.integers()))
    def test_repr(self, input_data):
        # Setup test
        container = Container(input_data)
        # Test action
        self.assertEqual(container.__repr__(), container.data.__repr__())

    @given(input_data=st.lists(st.integers()))
    def test_len(self, input_data):
        # Setup test
        container = Container(input_data)
        # Test action
        self.assertEqual(len(container), len(container.data))

    @given(input_data=st.lists(st.integers()))
    def test_reversed_where_reversed_reversed_x_equals_x(self, input_data):
        # Setup test
        container = Container(input_data)
        # Test action
        self.assertEqual(list(reversed(list(reversed(container)))), input_data)

    @given(x_data=st.lists(st.integers()), y_data=st.lists(st.integers()))
    def test_reversed_where_xy_equals_reversed_y_plus_reversed_x(self, x_data, y_data):
        # Setup test
        xy_reversed = list(reversed(Container(x_data + y_data)))
        x_reversed = list(reversed(Container(x_data)))
        y_reversed = list(reversed(Container(y_data)))
        # Test action
        self.assertEqual(xy_reversed, y_reversed + x_reversed)


if __name__ == '__main__':
    unittest.main()

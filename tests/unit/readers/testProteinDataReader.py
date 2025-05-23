import unittest
from typing import List
from unittest.mock import mock_open, patch

from chebai.preprocessing.reader import EMBEDDING_OFFSET

from chebai_proteins.preprocessing.reader import ProteinDataReader


class TestProteinDataReader(unittest.TestCase):
    """
    Unit tests for the ProteinDataReader class.
    """

    @classmethod
    @patch(
        "chebai.preprocessing.reader.open",
        new_callable=mock_open,
        read_data="M\nK\nT\nF\nR\nN",
    )
    def setUpClass(cls, mock_file: mock_open) -> None:
        """
        Set up the test environment by initializing a ProteinDataReader instance with a mocked token file.

        Args:
            mock_file: Mock object for file operations.
        """
        cls.reader = ProteinDataReader(token_path="/mock/path")
        # After initializing, cls.reader.cache should now be set to ['M', 'K', 'T', 'F', 'R', 'N']
        assert list(cls.reader.cache.items()) == list(
            {
                "M": 0,
                "K": 1,
                "T": 2,
                "F": 3,
                "R": 4,
                "N": 5,
            }.items()
        ), "Initial cache does not match expected values or the order doesn't match."

    def test_read_data(self) -> None:
        """
        Test the _read_data method with a protein sequence to ensure it correctly tokenizes the sequence.
        """
        raw_data = "MKTFFRN"

        # Expected output based on the cached tokens
        expected_output: List[int] = [
            EMBEDDING_OFFSET + 0,  # M
            EMBEDDING_OFFSET + 1,  # K
            EMBEDDING_OFFSET + 2,  # T
            EMBEDDING_OFFSET + 3,  # F
            EMBEDDING_OFFSET + 3,  # F (repeated token)
            EMBEDDING_OFFSET + 4,  # R
            EMBEDDING_OFFSET + 5,  # N
        ]
        result = self.reader._read_data(raw_data)
        self.assertEqual(
            result,
            expected_output,
            "The _read_data method did not produce the expected tokenized output.",
        )

    def test_read_data_with_new_token(self) -> None:
        """
        Test the _read_data method with a protein sequence that includes a new token.
        Ensure that the new token is added to the cache and processed correctly.
        """
        raw_data = "MKTFY"

        # 'Y' is not in the initial cache and should be added.
        expected_output: List[int] = [
            EMBEDDING_OFFSET + 0,  # M
            EMBEDDING_OFFSET + 1,  # K
            EMBEDDING_OFFSET + 2,  # T
            EMBEDDING_OFFSET + 3,  # F
            EMBEDDING_OFFSET + len(self.reader.cache),  # Y (new token)
        ]

        result = self.reader._read_data(raw_data)
        self.assertEqual(
            result,
            expected_output,
            "The _read_data method did not correctly handle a new token.",
        )

        # Verify that 'Y' was added to the cache
        self.assertIn(
            "Y", self.reader.cache, "The new token 'Y' was not added to the cache."
        )
        # Ensure it's at the correct index
        self.assertEqual(
            self.reader.cache["Y"],
            len(self.reader.cache) - 1,
            "The new token 'Y' was not added at the correct index in the cache.",
        )

    def test_read_data_with_invalid_token(self) -> None:
        """
        Test the _read_data method with an invalid amino acid token to ensure it raises a KeyError.
        """
        raw_data = "MKTFZ"  # 'Z' is not a valid amino acid token

        with self.assertRaises(KeyError) as context:
            self.reader._read_data(raw_data)

        self.assertIn(
            "Invalid token 'Z' encountered",
            str(context.exception),
            "The KeyError did not contain the expected message for an invalid token.",
        )

    def test_read_data_with_empty_sequence(self) -> None:
        """
        Test the _read_data method with an empty protein sequence to ensure it returns an empty list.
        """
        raw_data = ""

        result = self.reader._read_data(raw_data)
        self.assertEqual(
            result,
            [],
            "The _read_data method did not return an empty list for an empty input sequence.",
        )

    def test_read_data_with_repeated_tokens(self) -> None:
        """
        Test the _read_data method with repeated amino acid tokens to ensure it handles them correctly.
        """
        raw_data = "MMMMM"

        expected_output: List[int] = [EMBEDDING_OFFSET + 0] * 5  # All tokens are 'M'

        result = self.reader._read_data(raw_data)
        self.assertEqual(
            result,
            expected_output,
            "The _read_data method did not correctly handle repeated tokens.",
        )

    @patch("builtins.open", new_callable=mock_open)
    def test_finish_method_for_new_tokens(self, mock_file: mock_open) -> None:
        """
        Test the on_finish method to ensure it appends only the new tokens to the token file in order.
        """
        # Simulate that some tokens were already loaded
        self.reader._loaded_tokens_count = 6  # 6 tokens already loaded
        self.reader.cache = {
            "M": 0,
            "K": 1,
            "T": 2,
            "F": 3,
            "R": 4,
            "N": 5,
            "W": 6,  # New token 1
            "Y": 7,  # New token 2
            "V": 8,  # New token 3
            "Q": 9,  # New token 4
            "E": 10,  # New token 5
        }

        # Run the on_finish method
        self.reader.on_finish()

        # Check that the file was opened in append mode ('a')
        mock_file.assert_called_with(self.reader.token_path, "a")

        # Verify the new tokens were written in the correct order
        mock_file().writelines.assert_called_with(["W\n", "Y\n", "V\n", "Q\n", "E\n"])

    def test_finish_method_no_new_tokens(self) -> None:
        """
        Test the on_finish method when no new tokens are added (cache is the same).
        """
        self.reader._loaded_tokens_count = 6  # No new tokens
        self.reader.cache = {
            "M": 0,
            "K": 1,
            "T": 2,
            "F": 3,
            "R": 4,
            "N": 5,
        }

        with patch("builtins.open", new_callable=mock_open) as mock_file:
            self.reader.on_finish()
            # Check that no new tokens were written
            mock_file().writelines.assert_not_called()


if __name__ == "__main__":
    unittest.main()

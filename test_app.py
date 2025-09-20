""
Test script for FloodWise PH application.
"""
import sys
import os
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the components to test
from floodwise_ph.components.data.loader import FloodControlDataLoader
from floodwise_ph.components.data.processor import DataProcessor
from floodwise_ph.components.llm.base import LLMHandler

class TestDataLoader(unittest.TestCase):
    """Test cases for FloodControlDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = FloodControlDataLoader()
        self.sample_data = {
            'project_name': ['Project A', 'Project B', 'Project C'],
            'contractor': ['Contractor X', 'Contractor Y', 'Contractor Z'],
            'cost': [1000000, 2000000, 3000000],
            'location': ['Manila', 'Cebu', 'Davao']
        }
        self.df = pd.DataFrame(self.sample_data)
    
    def test_load_dataframe(self):
        """Test loading data from a pandas DataFrame."""
        result = self.loader.load_dataframe(self.df)
        self.assertTrue(result)
        self.assertEqual(len(self.loader.get_data()), 3)
        self.assertEqual(self.loader.get_available_columns(), ['project_name', 'contractor', 'cost', 'location'])
    
    def test_get_unique_values(self):
        """Test getting unique values for a column."""
        self.loader.load_dataframe(self.df)
        unique_contractors = self.loader.get_unique_values('contractor')
        self.assertEqual(len(unique_contractors), 3)
        self.assertIn('Contractor X', unique_contractors)


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = FloodControlDataLoader()
        self.sample_data = {
            'project_name': ['Project A', 'Project B', 'Project C', 'Project D'],
            'contractor': ['Contractor X', 'Contractor Y', 'Contractor X', 'Contractor Z'],
            'cost': [1000000, 2000000, 1500000, 3000000],
            'location': ['Manila', 'Cebu', 'Manila', 'Davao']
        }
        self.df = pd.DataFrame(self.sample_data)
        self.loader.load_dataframe(self.df)
        self.processor = DataProcessor(self.loader)
    
    def test_filter_data(self):
        """Test filtering data."""
        # Filter by contractor
        filtered = self.processor.filter_data({'contractor': 'Contractor X'})
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(filtered['contractor'] == 'Contractor X'))
        
        # Filter by location
        filtered = self.processor.filter_data({'location': 'Manila'})
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(filtered['location'] == 'Manila'))
    
    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        stats = self.processor.get_summary_stats('cost')
        self.assertEqual(stats['count'], 4)
        self.assertEqual(stats['min'], 1000000)
        self.assertEqual(stats['max'], 3000000)
        self.assertAlmostEqual(stats['mean'], 1875000.0)


class TestLLMHandler(unittest.TestCase):
    """Test cases for LLMHandler."""
    
    @patch('openai.OpenAI')
    def setUp(self, mock_openai):
        """Set up test fixtures."""
        # Mock the OpenAI client
        self.mock_client = MagicMock()
        mock_openai.return_value = self.mock_client
        
        # Mock the models.list() response
        mock_models = MagicMock()
        mock_models.data = [MagicMock(id='gpt-3.5-turbo')]
        self.mock_client.models.list.return_value = mock_models
        
        # Initialize the handler
        self.handler = LLMHandler()
    
    def test_is_available(self):
        """Test checking if the LLM service is available."""
        self.assertTrue(self.handler.is_available())
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_generate_response(self, mock_sleep):
        """Test generating a response from the LLM."""
        # Mock the chat completion response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "This is a test response."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Test generating a response
        response = self.handler.generate_response(
            query="What can you tell me about flood control projects?",
            context={"data": pd.DataFrame()}
        )
        
        self.assertEqual(response, "This is a test response.")
        self.mock_client.chat.completions.create.assert_called_once()


if __name__ == '__main__':
    unittest.main()

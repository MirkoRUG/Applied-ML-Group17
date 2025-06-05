"""
Test script to verify the CatBoost model implementation.
"""

import unittest
import pandas as pd
import sys
import os

# Add the project root to the path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from project_name.models.catboost_model import get_model, make_prediction


class TestCatBoostModel(unittest.TestCase):
    """Test cases for CatBoost model functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.test_data_single = {
            'name': 'Inception',
            'rating': 'PG-13',
            'genre': 'Sci-Fi',
            'year': 2010,
            'released': 'July 16, 2010 (United States)',
            'director': 'Christopher Nolan',
            'writer': 'Christopher Nolan',
            'star': 'Leonardo DiCaprio',
            'country': 'United States',
            'budget': 160000000,
            'company': 'Warner Bros.',
            'runtime': 148
        }
        
        cls.test_data_batch = pd.DataFrame([
            {
                'name': 'Movie A',
                'rating': 'PG',
                'genre': 'Comedy',
                'year': 2020,
                'released': 'January 1, 2020 (United States)',
                'director': 'Director A',
                'writer': 'Writer A',
                'star': 'Star A',
                'country': 'United States',
                'budget': 50000000,
                'company': 'Studio A',
                'runtime': 90
            },
            {
                'name': 'Movie B',
                'rating': 'R',
                'genre': 'Horror',
                'year': 2021,
                'released': 'October 31, 2021 (United States)',
                'director': 'Director B',
                'writer': 'Writer B',
                'star': 'Star B',
                'country': 'United States',
                'budget': 30000000,
                'company': 'Studio B',
                'runtime': 105
            }
        ])

    def test_get_model(self):
        """Test the get_model function."""
        model = get_model()
        self.assertIsNotNone(model, "Model should be loaded successfully")
        self.assertTrue(hasattr(model, 'predict'), "Model should have predict method")

    def test_make_prediction_single(self):
        """Test the make_prediction function with single input."""
        prediction = make_prediction(self.test_data_single)
        
        self.assertIsInstance(prediction, float, "Prediction should be a float")
        self.assertGreater(prediction, 0, "Prediction should be positive")
        self.assertLess(prediction, 10, "Prediction should be reasonable (< 10)")

    def test_model_predict_single(self):
        """Test the model.predict method with single input."""
        model = get_model()
        prediction = model.predict(self.test_data_single)
        
        self.assertIsInstance(prediction, float, "Prediction should be a float")
        self.assertGreater(prediction, 0, "Prediction should be positive")
        self.assertLess(prediction, 10, "Prediction should be reasonable (< 10)")

    def test_model_predict_batch(self):
        """Test batch prediction with multiple movies."""
        model = get_model()
        predictions = model.predict(self.test_data_batch)
        
        self.assertEqual(len(predictions), 2, "Should return 2 predictions for 2 movies")
        for prediction in predictions:
            self.assertIsInstance(prediction, (float, int), "Each prediction should be numeric")
            self.assertGreater(prediction, 0, "Each prediction should be positive")
            self.assertLess(prediction, 10, "Each prediction should be reasonable (< 10)")

    def test_prediction_consistency(self):
        """Test that predictions are consistent across multiple calls."""
        prediction1 = make_prediction(self.test_data_single)
        prediction2 = make_prediction(self.test_data_single)
        
        self.assertAlmostEqual(prediction1, prediction2, places=5, 
                              msg="Predictions should be consistent across calls")

    def test_model_with_missing_optional_fields(self):
        """Test model with missing optional fields."""
        minimal_data = {
            'name': 'Test Movie',
            'rating': 'PG-13',
            'genre': 'Drama',
            'released': 'January 1, 2020 (United States)',
            'director': 'Test Director',
            'writer': 'Test Writer',
            'star': 'Test Star',
            'country': 'United States',
            'company': 'Test Studio'
            # Missing: year, budget, runtime
        }
        
        # Should not raise an exception
        prediction = make_prediction(minimal_data)
        self.assertIsInstance(prediction, float, "Should handle missing optional fields")


def run_interactive_tests():
    """Run tests with detailed output for manual verification."""
    print("=" * 60)
    print("CatBoost Model Implementation - Interactive Tests")
    print("=" * 60)
    
    test_data = {
        'name': 'Inception',
        'rating': 'PG-13',
        'genre': 'Sci-Fi',
        'year': 2010,
        'released': 'July 16, 2010 (United States)',
        'director': 'Christopher Nolan',
        'writer': 'Christopher Nolan',
        'star': 'Leonardo DiCaprio',
        'country': 'United States',
        'budget': 160000000,
        'company': 'Warner Bros.',
        'runtime': 148
    }
    
    print("\n1. Testing get_model function...")
    try:
        model = get_model()
        print("✓ get_model function works correctly")
    except Exception as e:
        print(f"✗ get_model function failed: {e}")
        return
    
    print("\n2. Testing make_prediction function...")
    try:
        prediction = make_prediction(test_data)
        print(f"✓ make_prediction function works correctly")
        print(f"  Predicted score for Inception: {prediction:.2f}")
    except Exception as e:
        print(f"✗ make_prediction function failed: {e}")
    
    print("\n3. Testing model.predict method...")
    try:
        prediction = model.predict(test_data)
        print(f"✓ model.predict method works correctly")
        print(f"  Predicted score for Inception: {prediction:.2f}")
    except Exception as e:
        print(f"✗ model.predict method failed: {e}")
    
    print("\n4. Testing batch prediction...")
    try:
        batch_data = pd.DataFrame([
            {**test_data, 'name': 'Movie A', 'genre': 'Comedy'},
            {**test_data, 'name': 'Movie B', 'genre': 'Horror'}
        ])
        predictions = model.predict(batch_data)
        print(f"✓ Batch prediction works correctly")
        print(f"  Predicted scores: {[f'{p:.2f}' for p in predictions]}")
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Interactive tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CatBoost model tests')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run interactive tests with detailed output')
    parser.add_argument('--unittest', action='store_true', 
                       help='Run unit tests')
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_tests()
    elif args.unittest or len(sys.argv) == 1:
        # Run unit tests by default
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        print("Use --interactive for detailed output or --unittest for unit tests")

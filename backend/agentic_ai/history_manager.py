"""
User History Manager
Stores and manages test history using JSON file storage
Can be easily adapted to use database (SQLite, Firestore, etc.)
"""

import json
import os
from datetime import datetime
from pathlib import Path
import config


class HistoryManager:
    """
    Manages user test history with persistence
    """
    
    def __init__(self, storage_path=None, user_id='default'):
        """
        Initialize history manager
        
        Args:
            storage_path: Path to store history files
            user_id: User identifier for multi-user support
        """
        self.storage_path = storage_path or (config.BASE_DIR / 'data' / 'history')
        self.storage_path = Path(self.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.user_id = user_id
        self.history_file = self.storage_path / f'{user_id}_history.json'
        
        self.history = self._load_history()
    
    def _load_history(self):
        """Load history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history: {e}")
                return []
        return []
    
    def _save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
            return False
    
    def save_test(self, test_result):
        """
        Save a new test result
        
        Args:
            test_result: Dict containing test information
            
        Returns:
            bool: Success status
        """
        # Ensure required fields
        if 'date' not in test_result:
            test_result['date'] = datetime.now().isoformat()
        
        if 'id' not in test_result:
            test_result['id'] = f"VPX-{int(datetime.now().timestamp())}"
        
        # Add to history
        self.history.insert(0, test_result)  # Most recent first
        
        # Save to file
        return self._save_history()
    
    def get_all_tests(self, limit=None, sort_desc=True):
        """
        Get all test results
        
        Args:
            limit: Maximum number of results to return
            sort_desc: Sort by date descending (most recent first)
            
        Returns:
            list: Test results
        """
        tests = self.history.copy()
        
        if sort_desc:
            tests.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        if limit:
            tests = tests[:limit]
        
        return tests
    
    def get_test_by_id(self, test_id):
        """
        Get specific test by ID
        
        Args:
            test_id: Test ID
            
        Returns:
            dict or None: Test result
        """
        for test in self.history:
            if test.get('id') == test_id:
                return test
        return None
    
    def get_latest_test(self):
        """
        Get most recent test
        
        Returns:
            dict or None: Latest test result
        """
        if self.history:
            return max(self.history, key=lambda x: x.get('date', ''))
        return None
    
    def get_tests_in_range(self, start_date, end_date):
        """
        Get tests within date range
        
        Args:
            start_date: Start date (ISO format string or datetime)
            end_date: End date (ISO format string or datetime)
            
        Returns:
            list: Tests within range
        """
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()
        if isinstance(end_date, datetime):
            end_date = end_date.isoformat()
        
        return [
            test for test in self.history
            if start_date <= test.get('date', '') <= end_date
        ]
    
    def get_trend_data(self, num_tests=5):
        """
        Get trend data for visualization
        
        Args:
            num_tests: Number of recent tests to include
            
        Returns:
            dict: Trend data with dates and scores
        """
        recent_tests = self.get_all_tests(limit=num_tests, sort_desc=True)
        recent_tests.reverse()  # Chronological order for trends
        
        return {
            'dates': [test.get('date', '') for test in recent_tests],
            'risk_scores': [test.get('risk_score', 0) for test in recent_tests],
            'confidence': [test.get('confidence', 0) for test in recent_tests],
            'risk_levels': [test.get('risk_level', 'Unknown') for test in recent_tests]
        }
    
    def calculate_statistics(self):
        """
        Calculate statistics across all tests
        
        Returns:
            dict: Statistical summary
        """
        if not self.history:
            return {
                'total_tests': 0,
                'average_risk': 0,
                'trend': 'insufficient_data'
            }
        
        risk_scores = [t.get('risk_score', 0) for t in self.history]
        
        stats = {
            'total_tests': len(self.history),
            'average_risk': sum(risk_scores) / len(risk_scores),
            'min_risk': min(risk_scores),
            'max_risk': max(risk_scores),
            'std_dev': self._calculate_std(risk_scores)
        }
        
        # Calculate trend
        if len(self.history) >= 2:
            recent = sorted(self.history, key=lambda x: x.get('date', ''), reverse=True)
            latest_3 = [t.get('risk_score', 0) for t in recent[:3]]
            older_3 = [t.get('risk_score', 0) for t in recent[-3:]]
            
            latest_avg = sum(latest_3) / len(latest_3)
            older_avg = sum(older_3) / len(older_3)
            
            if latest_avg < older_avg - 0.05:
                stats['trend'] = 'improving'
            elif latest_avg > older_avg + 0.05:
                stats['trend'] = 'worsening'
            else:
                stats['trend'] = 'stable'
        else:
            stats['trend'] = 'insufficient_data'
        
        return stats
    
    def _calculate_std(self, values):
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def delete_test(self, test_id):
        """
        Delete a test by ID
        
        Args:
            test_id: Test ID to delete
            
        Returns:
            bool: Success status
        """
        original_length = len(self.history)
        self.history = [t for t in self.history if t.get('id') != test_id]
        
        if len(self.history) < original_length:
            return self._save_history()
        return False
    
    def clear_all_history(self):
        """
        Clear all test history (use with caution)
        
        Returns:
            bool: Success status
        """
        self.history = []
        return self._save_history()
    
    def export_to_dict(self):
        """
        Export all data as dictionary
        
        Returns:
            dict: Complete history data
        """
        return {
            'user_id': self.user_id,
            'total_tests': len(self.history),
            'statistics': self.calculate_statistics(),
            'tests': self.history
        }
    
    def import_from_dict(self, data):
        """
        Import history from dictionary
        
        Args:
            data: History data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            if 'tests' in data:
                self.history = data['tests']
                return self._save_history()
            return False
        except Exception as e:
            print(f"Error importing history: {e}")
            return False


# Utility functions
def get_user_history(user_id='default', storage_path=None):
    """
    Quick utility to get user history
    
    Args:
        user_id: User identifier
        storage_path: Optional storage path
        
    Returns:
        list: User's test history
    """
    manager = HistoryManager(storage_path, user_id)
    return manager.get_all_tests()


def save_test_result(test_result, user_id='default', storage_path=None):
    """
    Quick utility to save a test result
    
    Args:
        test_result: Test data dictionary
        user_id: User identifier
        storage_path: Optional storage path
        
    Returns:
        bool: Success status
    """
    manager = HistoryManager(storage_path, user_id)
    return manager.save_test(test_result)


def get_trend_summary(user_id='default', num_tests=5):
    """
    Quick utility to get trend summary
    
    Args:
        user_id: User identifier
        num_tests: Number of recent tests
        
    Returns:
        dict: Trend data
    """
    manager = HistoryManager(user_id=user_id)
    return manager.get_trend_data(num_tests)

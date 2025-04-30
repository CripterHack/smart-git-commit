"""Tests for commit_group.py module."""

import unittest
from smart_git_commit.commit_group import CommitType, Change, CommitGroup

class TestCommitType(unittest.TestCase):
    """Test cases for CommitType enum."""
    
    def test_commit_type_values(self):
        """Test commit type values."""
        self.assertEqual(str(CommitType.FEAT), "feat")
        self.assertEqual(str(CommitType.FIX), "fix")
        self.assertEqual(str(CommitType.DOCS), "docs")
        self.assertEqual(str(CommitType.STYLE), "style")
        self.assertEqual(str(CommitType.REFACTOR), "refactor")
        self.assertEqual(str(CommitType.PERF), "perf")
        self.assertEqual(str(CommitType.TEST), "test")
        self.assertEqual(str(CommitType.BUILD), "build")
        self.assertEqual(str(CommitType.CI), "ci")
        self.assertEqual(str(CommitType.CHORE), "chore")
        self.assertEqual(str(CommitType.REVERT), "revert")

class TestChange(unittest.TestCase):
    """Test cases for Change class."""
    
    def test_change_creation(self):
        """Test creating a Change object."""
        change = Change("test.py", "M", "backend")
        self.assertEqual(change.filename, "test.py")
        self.assertEqual(change.status, "M")
        self.assertEqual(change.component, "backend")
        self.assertEqual(change.content_diff, "")
    
    def test_change_with_diff(self):
        """Test creating a Change object with diff content."""
        diff = "- old line\n+ new line"
        change = Change("test.py", "M", "backend", diff)
        self.assertEqual(change.content_diff, diff)
    
    def test_change_str(self):
        """Test string representation of Change."""
        change = Change("test.py", "M", "backend")
        self.assertEqual(str(change), "M test.py")

class TestCommitGroup(unittest.TestCase):
    """Test cases for CommitGroup class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.changes = [
            Change("src/test.py", "M", "backend"),
            Change("src/other.py", "A", "backend")
        ]
    
    def test_commit_group_creation(self):
        """Test creating a CommitGroup object."""
        group = CommitGroup(self.changes, CommitType.FEAT)
        self.assertEqual(group.changes, self.changes)
        self.assertEqual(group.commit_type, CommitType.FEAT)
        self.assertEqual(group.component, "backend")  # Auto-detected from changes
    
    def test_commit_group_with_scope(self):
        """Test CommitGroup with scope."""
        group = CommitGroup(
            self.changes,
            CommitType.FEAT,
            scope="api",
            description="Add new API endpoint"
        )
        self.assertEqual(group.scope, "api")
        self.assertEqual(group.description, "Add new API endpoint")
    
    def test_commit_group_breaking_change(self):
        """Test CommitGroup with breaking change."""
        group = CommitGroup(
            self.changes,
            CommitType.FEAT,
            breaking=True,
            description="Breaking API change"
        )
        self.assertTrue(group.breaking)
        self.assertIn("!", str(group))
    
    def test_commit_group_with_issues(self):
        """Test CommitGroup with issue references."""
        group = CommitGroup(
            self.changes,
            CommitType.FIX,
            description="Fix bug",
            issues={"#123", "#456"}
        )
        self.assertEqual(group.issues, {"#123", "#456"})
        message = str(group)
        self.assertIn("#123", message)
        self.assertIn("#456", message)
    
    def test_commit_group_auto_component(self):
        """Test automatic component detection."""
        # Single component
        group = CommitGroup(self.changes, CommitType.FEAT)
        self.assertEqual(group.component, "backend")
        
        # Multiple components
        mixed_changes = [
            Change("src/test.py", "M", "backend"),
            Change("docs/README.md", "M", "docs")
        ]
        group = CommitGroup(mixed_changes, CommitType.FEAT)
        self.assertEqual(group.component, "")  # No single component
    
    def test_commit_group_str_minimal(self):
        """Test minimal string representation."""
        group = CommitGroup(self.changes, CommitType.FEAT)
        message = str(group)
        self.assertTrue(message.startswith("feat"))
        self.assertIn("M src/test.py", message)
        self.assertIn("A src/other.py", message)
    
    def test_commit_group_str_full(self):
        """Test full string representation."""
        group = CommitGroup(
            self.changes,
            CommitType.FEAT,
            scope="api",
            description="Add new endpoint",
            body="Detailed description\nMultiple lines",
            breaking=True,
            issues={"#123"}
        )
        message = str(group)
        
        # Check all parts are present
        self.assertTrue(message.startswith("feat(api)!: Add new endpoint"))
        self.assertIn("Detailed description", message)
        self.assertIn("Multiple lines", message)
        self.assertIn("Affected files:", message)
        self.assertIn("M src/test.py", message)
        self.assertIn("A src/other.py", message)
        self.assertIn("#123", message)

if __name__ == '__main__':
    unittest.main() 
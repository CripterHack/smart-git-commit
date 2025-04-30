"""Commit group and type definitions."""

from enum import Enum
from typing import List, Set
from dataclasses import dataclass, field

# Import GitChange instead (adjust path if needed, assuming it's in smart_git_commit.py)
from .smart_git_commit import GitChange

class CommitType(Enum):
    """Conventional commit types."""
    FEAT = "feat"
    FIX = "fix"
    DOCS = "docs"
    STYLE = "style"
    REFACTOR = "refactor"
    PERF = "perf"
    TEST = "test"
    BUILD = "build"
    CI = "ci"
    CHORE = "chore"
    REVERT = "revert"
    
    def __str__(self) -> str:
        """Return the commit type value."""
        return self.value

@dataclass
class CommitGroup:
    """Group of related changes to be committed together."""
    changes: List[GitChange]
    commit_type: CommitType
    component: str = ""  # Main component affected by changes
    scope: str = ""  # Optional scope for conventional commits
    description: str = ""  # Short description of changes
    body: str = ""  # Detailed description of changes
    breaking: bool = False  # Whether this is a breaking change
    issues: Set[str] = field(default_factory=set)  # Related issue references
    
    def __post_init__(self):
        """Post-initialization processing."""
        # If no component is specified, try to determine from changes
        if not self.component and self.changes:
            components = {change.component for change in self.changes}
            if len(components) == 1:
                self.component = next(iter(components))
    
    def __str__(self) -> str:
        """Return a string representation of the commit group."""
        parts = []
        
        # Add commit type
        parts.append(str(self.commit_type))
        
        # Add scope if present
        if self.scope:
            parts.append(f"({self.scope})")
        
        # Add breaking change marker
        if self.breaking:
            parts.append("!")
        
        # Add description
        if self.description:
            parts.append(f": {self.description}")
        
        # Add body if present
        message = "".join(parts)
        if self.body:
            message += f"\n\n{self.body}"
        
        # Add affected files
        if self.changes:
            message += "\n\nAffected files:"
            for change in self.changes:
                # Determine symbol based on CLEANED status
                if change.status == "??": status_symbol = "+"
                elif change.status == "D": status_symbol = "-"
                else: status_symbol = "M" # Default for M, A, R, C
                
                # Ensure clean filename (remove any status prefix)
                clean_filename = change.filename
                if clean_filename.startswith(('M ', 'A ', 'D ', '?? ')):
                    clean_filename = clean_filename[2:].strip()
                elif clean_filename.startswith(('M', 'A', 'D', '??')) and len(clean_filename) > 1:
                    if clean_filename[1] != ' ':
                        clean_filename = clean_filename[1:].strip()
                
                message += f"\n- {status_symbol} {clean_filename}"
        
        # Add issue references
        if self.issues:
            message += f"\n\n{' '.join(sorted(self.issues))}"
        
        return message 
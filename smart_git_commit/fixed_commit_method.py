from typing import List

class GitCommit:
    def _determine_commit_type(self, changes: List[GitChange]) -> CommitType:
        """Determine the commit type based on the changes."""
        # Count filenames matching different patterns
        patterns = {
            CommitType.FEAT: ['feature', 'feat', 'add', 'new'],
            CommitType.FIX: ['fix', 'bug', 'issue', 'hotfix', 'patch'],
            CommitType.DOCS: ['doc', 'readme', '.md'],
            CommitType.STYLE: ['style', 'format', 'lint', '.css', '.scss'],
            CommitType.REFACTOR: ['refactor', 'clean', 'rewrite'],
            CommitType.TEST: ['test', 'spec', 'check'],
            CommitType.CHORE: ['chore', 'build', 'ci', 'tooling'],
            CommitType.PERF: ['perf', 'performance', 'optimize', 'speed']
        }
        
        # Count matches for each pattern
        counts = {commit_type: 0 for commit_type in CommitType}
        
        for change in changes:
            for commit_type, keywords in patterns.items():
                for keyword in keywords:
                    if keyword.lower() in change.filename.lower():
                        counts[commit_type] += 1
                        break
        
        # Check if any files were deleted
        has_deletions = any(change.status.startswith('D') for change in changes)
        
        # Use the most common commit type, or default to CHORE
        if has_deletions and counts[CommitType.FEAT] == 0:
            return CommitType.REFACTOR  # Deletions are usually refactoring
        
        if max(counts.values()) > 0:
            # Get the commit type with the highest count
            return max(counts.items(), key=lambda x: x[1])[0]
        
        # Default to FEAT for new files, otherwise CHORE
        if any(change.status.startswith('A') for change in changes):
            return CommitType.FEAT
        
        return CommitType.CHORE 
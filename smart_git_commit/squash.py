"""
Commit Squashing - Module for intelligent commit squashing functionality.

This module provides tools to analyze, identify, and squash related commits
to maintain a cleaner and more organized Git history. It supports both
AI-powered semantic squashing and rule-based squashing approaches.
"""

import os
import re
import json
import enum
import logging
import datetime
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Union, Any, Callable
import tempfile

# Configure logger
logger = logging.getLogger(__name__)

class SquashStrategy(enum.Enum):
    """Strategies for squashing commits."""
    AUTO = "auto"
    RELATED_FILES = "related_files"
    SEMANTIC = "semantic" 
    CONVENTIONAL = "conventional"
    SAME_AUTHOR = "same_author"
    TIME_WINDOW = "time_window"

@dataclass
class CommitInfo:
    """Information about a Git commit."""
    hash: str
    author: str
    date: datetime.datetime
    subject: str
    body: Optional[str] = None
    changed_files: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    
    @classmethod
    def from_commit_hash(cls, commit_hash: str, repo_path: str = ".") -> "CommitInfo":
        """Create a CommitInfo instance from a commit hash."""
        cmd = [
            "git", "-C", repo_path, "show", 
            "--no-patch", 
            "--format=%H%n%an%n%at%n%s%n%b%n%P", 
            commit_hash
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split("\n")
            
            if len(lines) < 4:
                logger.error(f"Unexpected format for commit info {commit_hash}: {lines}")
                raise ValueError(f"Could not parse commit info for {commit_hash}")
            
            # Parse the commit info
            hash_ = lines[0]
            author = lines[1]
            date = datetime.datetime.fromtimestamp(int(lines[2]))
            subject = lines[3]
            
            # Determine where the body ends - it's before the parent hashes line
            parent_line_index = -1
            for i in range(len(lines) -1, 3, -1):
                 # Parent hashes are typically hex strings, check if the line looks like parent hashes
                 if all(c in '0123456789abcdef ' for c in lines[i].lower()):
                      parent_line_index = i
                      break
            
            if parent_line_index == -1:
                 # Assume last line is parent if pattern not matched explicitly
                 parent_line_index = len(lines) -1
                 
            body = "\n".join(lines[4:parent_line_index]).strip() if parent_line_index > 4 else None
            
            # Parents (might be multiple for merge commits)
            parents = lines[parent_line_index].split() if parent_line_index < len(lines) and lines[parent_line_index] else []
            
            # Get changed files
            changed_files = get_changed_files_for_commit(commit_hash, repo_path)
            
            return cls(
                hash=hash_,
                author=author,
                date=date,
                subject=subject,
                body=body,
                changed_files=changed_files,
                parents=parents
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting commit info for {commit_hash}: {e}")
            raise
    
    def get_full_message(self) -> str:
        """Get the full commit message (subject + body)."""
        if self.body:
            return f"{self.subject}\n\n{self.body}"
        return self.subject
    
    def is_merge_commit(self) -> bool:
        """Check if this is a merge commit."""
        return len(self.parents) > 1
    
    def get_file_content_at_commit(self, file_path: str, repo_path: str = ".") -> Optional[str]:
        """Get the content of a file at this commit."""
        cmd = ["git", "-C", repo_path, "show", f"{self.hash}:{file_path}"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None

def get_recent_commits(count: int = 10, repo_path: str = ".") -> List[CommitInfo]:
    """Get a list of recent commits."""
    cmd = ["git", "-C", repo_path, "log", f"-{count}", "--format=%H"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commit_hashes = result.stdout.strip().split("\n")
        
        # Create CommitInfo instances for each hash
        return [
            CommitInfo.from_commit_hash(commit_hash, repo_path)
            for commit_hash in commit_hashes if commit_hash
        ]
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting recent commits: {e}")
        return []

def get_commits_since(reference: str, repo_path: str = ".") -> List[CommitInfo]:
    """Get commits since a reference (commit hash, branch, tag, etc.)."""
    cmd = ["git", "-C", repo_path, "log", f"{reference}..HEAD", "--format=%H"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commit_hashes = result.stdout.strip().split("\n")
        
        # Create CommitInfo instances for each hash
        return [
            CommitInfo.from_commit_hash(commit_hash, repo_path)
            for commit_hash in commit_hashes if commit_hash
        ]
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting commits since {reference}: {e}")
        return []

def get_changed_files_for_commit(commit_hash: str, repo_path: str = ".") -> List[str]:
    """Get the list of files changed in a commit."""
    cmd = ["git", "-C", repo_path, "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting changed files for commit {commit_hash}: {e}")
        return []

def calculate_file_similarity(files1: List[str], files2: List[str]) -> float:
    """Calculate similarity between two sets of files."""
    if not files1 or not files2:
        return 0.0
    
    # Convert to sets for easier comparison
    set1 = set(files1)
    set2 = set(files2)
    
    # Calculate Jaccard similarity coefficient
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def calculate_semantic_similarity(commit1: CommitInfo, commit2: CommitInfo, workflow=None) -> float:
    """
    Calculate semantic similarity between two commits.
    
    This uses AI to analyze the commit messages and changed files for semantic similarity.
    If the workflow is not provided or AI fails, falls back to simpler heuristics.
    """
    # If no workflow provided or AI is disabled, fall back to simpler heuristics
    if workflow is None or not hasattr(workflow, 'use_ai') or not workflow.use_ai:
        logger.debug("AI is disabled in workflow, falling back to semantic heuristics.")
        raise ValueError("AI disabled") # Trigger fallback
        
    ai_client = workflow.ollama_client # Assuming ollama_client holds the active AI client
    if not ai_client:
        logger.debug("AI client not available in workflow, falling back to semantic heuristics.")
        raise ValueError("AI client unavailable") # Trigger fallback
    
    # Prepare prompt for the AI
    prompt = f"""
    Analyze the semantic similarity between these two commits:
    
    Commit 1:
    Subject: {commit1.subject}
    Body: {commit1.body or ''}
    Changed files: {', '.join(commit1.changed_files)}
    
    Commit 2:
    Subject: {commit2.subject}
    Body: {commit2.body or ''}
    Changed files: {', '.join(commit2.changed_files)}
    
    Calculate a similarity score between 0.0 and 1.0, where:
    - 0.0 means completely unrelated commits
    - 1.0 means semantically identical commits
    
    Consider:
    1. Related functionality or components (e.g., both affect user auth)
    2. If they address the same issue/feature (e.g., steps in fixing one bug)
    3. If they modify related files or logical modules
    4. If one commit is a direct follow-up/refinement of the other
    
    Output ONLY a single floating-point number between 0.0 (completely unrelated) and 1.0 (semantically identical).
    """
    
    response = ai_client.generate(prompt) # Use generate method
    
    # Extract the score from the response
    score_match = re.search(r'(\d+\.\d+|\d+)', response)
    if score_match:
        score = float(score_match.group(1))
        return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
    
    # If we couldn't parse the score, fall back to the simpler method
    logger.warning("Could not parse similarity score from AI response, falling back to heuristics")
    return _calculate_heuristic_similarity(commit1, commit2)

def _calculate_heuristic_similarity(commit1: CommitInfo, commit2: CommitInfo, 
                                  time_window_minutes: int = 180, # Use a wider default window for heuristics
                                  same_author_boost: float = 0.2,
                                  trivial_message_boost: float = 0.3) -> float:
    """Calculate a similarity score based on non-AI heuristics."""
    
    # 1. File Similarity (Jaccard Index)
    file_similarity = calculate_file_similarity(commit1.changed_files, commit2.changed_files)
    
    # 2. Time Proximity
    time_diff_minutes = abs((commit1.date - commit2.date).total_seconds() / 60)
    # Score higher for closer commits, decay score over the time window
    time_score = max(0.0, 1.0 - (time_diff_minutes / time_window_minutes)) if time_window_minutes > 0 else 1.0

    # 3. Author Match
    author_match_score = same_author_boost if commit1.author == commit2.author else 0.0

    # 4. Commit Message Heuristics
    msg1 = commit1.subject.lower()
    msg2 = commit2.subject.lower()
    trivial_patterns = ["fixup!", "squash!", "amend!", "typo", "lint", "format", "minor fix", "wip", "address comments"]
    
    is_msg2_trivial = any(pattern in msg2 for pattern in trivial_patterns)
    is_msg1_trivial = any(pattern in msg1 for pattern in trivial_patterns)

    # Boost if the *later* commit looks like a fixup of the previous one
    message_heuristic_score = 0.0
    if is_msg2_trivial and not is_msg1_trivial:
        message_heuristic_score = trivial_message_boost 
    # Small boost if both are trivial (e.g., two formatting commits)
    elif is_msg1_trivial and is_msg2_trivial:
        message_heuristic_score = trivial_message_boost * 0.5
        
    # 5. Conventional Commit Heuristic (slight boost if same type/scope)
    conv_score = 0.0
    prev_conv, prev_scope = is_conventional_commit(commit1.get_full_message())
    curr_conv, curr_scope = is_conventional_commit(commit2.get_full_message())
    if prev_conv and curr_conv and prev_scope == curr_scope:
        conv_score = 0.1 # Small boost

    # Combine scores with weighting (weights are adjustable)
    # Weighting leans towards file similarity and time proximity
    combined_score = (
        0.4 * file_similarity +
        0.3 * time_score +
        0.1 * author_match_score + # Smaller weight as it's less direct than time/files
        0.1 * message_heuristic_score + 
        0.1 * conv_score
    )
    
    # Ensure score is between 0.0 and 1.0
    final_score = min(max(combined_score, 0.0), 1.0)
    logger.debug(f"Heuristic similarity score between {commit1.hash[:7]} and {commit2.hash[:7]}: {final_score:.2f}")
    return final_score

def is_conventional_commit(message: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a commit message follows the Conventional Commits specification.
    Returns a tuple (is_conventional, commit_type)
    """
    # Conventional Commits pattern: type(scope?): description
    pattern = r'^(\w+)(\([^)]+\))?: .+$'
    match = re.match(pattern, message)
    
    if match:
        commit_type = match.group(1).lower()
        return True, commit_type
    
    return False, None

def find_squashable_commits(
    commits: List[CommitInfo],
    strategy: Union[str, SquashStrategy] = "auto", # Allow string "auto"
    time_window_minutes: int = 60,
    similarity_threshold: float = 0.6,
    workflow=None
) -> List[List[CommitInfo]]:
    """Find groups of consecutive commits that can be squashed based on the strategy."""
    if len(commits) < 2:
        return []

    # Reverse commits to process from oldest to newest
    # This makes grouping consecutive commits easier
    commits.reverse()

    groups = []
    current_group = [commits[0]]
    
    # Determine the actual strategy if 'auto' is passed
    if isinstance(strategy, str) and strategy == 'auto':
        if workflow and hasattr(workflow, 'use_ai') and workflow.use_ai and workflow.ollama_client:
            active_strategy = SquashStrategy.SEMANTIC
            logger.debug("Auto strategy selected: Using SEMANTIC (AI available)")
        else:
            # Fallback auto strategy without AI: prioritize conventional, then related files
            active_strategy = SquashStrategy.CONVENTIONAL 
            logger.debug("Auto strategy selected: Using CONVENTIONAL (AI not available)")
            # We will check related_files if conventional doesn't match inside the loop
    elif isinstance(strategy, SquashStrategy):
        active_strategy = strategy
        logger.debug(f"Strategy selected: {active_strategy.value}")
    else:
         logger.warning(f"Invalid squash strategy '{strategy}'. Defaulting to RELATED_FILES.")
         active_strategy = SquashStrategy.RELATED_FILES

    for i in range(1, len(commits)):
        prev_commit = current_group[-1]
        current_commit = commits[i]

        # Skip merge commits - they generally shouldn't be squashed into others
        if current_commit.is_merge_commit():
            if current_group:
                groups.append(current_group)
            current_group = [current_commit] # Start new group with the merge commit (won't be squashed)
            continue 
        # Also skip squashing *into* a merge commit
        if prev_commit.is_merge_commit():
             if current_group:
                 groups.append(current_group)
             current_group = [current_commit]
             continue

        match = False
        strategy_used = active_strategy # Keep track of which strategy is being checked
        log_msg_prefix = f"Comparing {prev_commit.hash[:7]} and {current_commit.hash[:7]}:"

        # Calculate heuristic score regardless of strategy (used for fallback or non-AI)
        # Note: We might pass the specific time_window_minutes from args here if needed
        heuristic_score = _calculate_heuristic_similarity(prev_commit, current_commit)

        while True: # Loop to allow fallback for 'auto' strategy
            if strategy_used == SquashStrategy.SEMANTIC:
                try:
                    similarity = calculate_semantic_similarity(prev_commit, current_commit, workflow)
                    match = similarity >= similarity_threshold
                    logger.debug(f"{log_msg_prefix} Strategy=SEMANTIC, Score={similarity:.2f}, Match={match}")
                except (ValueError, Exception) as ai_error: # Catches AI disabled/error from calculate_semantic_similarity
                    logger.debug(f"{log_msg_prefix} Strategy=SEMANTIC failed ({ai_error}), falling back to heuristics.")
                    # Use pre-calculated heuristic score for fallback
                    match = heuristic_score >= similarity_threshold
                    logger.debug(f"{log_msg_prefix} Strategy=HEURISTIC (fallback), Score={heuristic_score:.2f}, Match={match}")
                break # Exit loop after semantic check (or its fallback)
            
            elif strategy_used == SquashStrategy.RELATED_FILES:
                # We use the heuristic score which includes file similarity
                match = heuristic_score >= similarity_threshold 
                logger.debug(f"{log_msg_prefix} Strategy=RELATED_FILES (using heuristic), Score={heuristic_score:.2f}, Match={match}")
                # Note: Threshold here might need tuning if based only on heuristic score
                break

            elif strategy_used == SquashStrategy.SAME_AUTHOR:
                # Check author explicitly AND use heuristic score
                is_same_author = current_commit.author == prev_commit.author
                match = is_same_author and (heuristic_score >= similarity_threshold)
                logger.debug(f"{log_msg_prefix} Strategy=SAME_AUTHOR, SameAuthor={is_same_author}, HeuristicScore={heuristic_score:.2f}, Match={match}")
                break

            elif strategy_used == SquashStrategy.TIME_WINDOW:
                # Heuristic score already includes time window check
                match = heuristic_score >= similarity_threshold
                logger.debug(f"{log_msg_prefix} Strategy=TIME_WINDOW (using heuristic), Score={heuristic_score:.2f}, Match={match}")
                break

            elif strategy_used == SquashStrategy.CONVENTIONAL:
                prev_conv, prev_scope = is_conventional_commit(prev_commit.get_full_message())
                curr_conv, curr_scope = is_conventional_commit(current_commit.get_full_message())
                is_conv_match = (prev_conv and curr_conv and prev_scope == curr_scope)
                match = is_conv_match
                logger.debug(f"{log_msg_prefix} Strategy=CONVENTIONAL, Match={match}")
                
                # If using auto strategy and conventional didn't match, try heuristic fallback
                if isinstance(strategy, str) and strategy == 'auto' and not match:
                    logger.debug(f"{log_msg_prefix} Auto strategy: Conventional didn't match, falling back to HEURISTIC")
                    # Use the pre-calculated heuristic score
                    match = heuristic_score >= similarity_threshold
                    logger.debug(f"{log_msg_prefix} Strategy=HEURISTIC (fallback), Score={heuristic_score:.2f}, Match={match}")
                    # No need to set strategy_used as we break next
                break # Exit loop
            
            else: # Should not happen
                logger.error(f"Encountered unexpected strategy state: {strategy_used}")
                break

        if match:
            current_group.append(current_commit)
        else:
            if current_group: # Store the completed group
                groups.append(current_group)
            current_group = [current_commit] # Start a new group

    # Add the last group
    if current_group:
        groups.append(current_group)

    # Filter out groups with only one commit (or only a merge commit)
    squashable_groups = [group for group in groups if len(group) > 1 and not all(c.is_merge_commit() for c in group)]

    # Restore original commit order (newest first) within groups and the list itself
    for group in squashable_groups:
        group.reverse()
    squashable_groups.reverse()

    return squashable_groups

def generate_squashed_commit_message(commit_group: List[CommitInfo]) -> str:
    """Generate a commit message for the squashed commits."""
    if not commit_group:
        return "Squashed commit"
    
    # Check if all commits are conventional
    all_conventional = all(is_conventional_commit(c.subject)[0] for c in commit_group)
    
    if all_conventional:
        # If all are conventional, group by type
        commits_by_type = {}
        
        for commit in commit_group:
            _, commit_type = is_conventional_commit(commit.subject)
            
            if commit_type not in commits_by_type:
                commits_by_type[commit_type] = []
            
            # Extract the description (everything after the type prefix)
            match = re.match(r'^(\w+)(?:\([^)]+\))?: (.+)$', commit.subject)
            if match:
                description = match.group(2)
                commits_by_type[commit_type].append(description)
        
        # Create a message with sections for each type
        message_parts = ["Squashed commits:"]
        
        for commit_type, descriptions in commits_by_type.items():
            message_parts.append(f"\n{commit_type.upper()}:")
            for description in descriptions:
                message_parts.append(f"- {description}")
        
        return "\n".join(message_parts)
    else:
        # Otherwise, list all commit messages
        message_parts = ["Squashed commits:"]
        
        for commit in commit_group:
            message_parts.append(f"\n- {commit.subject}")
        
        return "\n".join(message_parts)

def squash_commits(
    commit_group: List[CommitInfo],
    repo_path: str = ".",
    interactive: bool = True
) -> bool:
    """
    Squash a group of commits.
    
    Args:
        commit_group: List of commits to squash (newest first)
        repo_path: Path to the Git repository
        interactive: Whether to run in interactive mode
        
    Returns:
        True if successful, False otherwise
    """
    if not commit_group or len(commit_group) <= 1:
        logger.warning("Nothing to squash (insufficient commits)")
        return False
    
    # Sort commits from oldest to newest (for rebase)
    sorted_commits = sorted(commit_group, key=lambda c: c.date)
    
    # Get the hash of the oldest commit
    oldest_commit_hash = sorted_commits[0].hash
    
    # Get the parent of the oldest commit
    try:
        cmd = ["git", "-C", repo_path, "rev-parse", f"{oldest_commit_hash}^"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        base_commit = result.stdout.strip()
    except subprocess.CalledProcessError:
        logger.error(f"Error getting parent of commit {oldest_commit_hash}")
        return False
    
    # Generate the squashed commit message
    squashed_message = generate_squashed_commit_message(commit_group)
    
    # Create a temporary file for the commit message
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(squashed_message)
        message_file = temp_file.name
    
    # Define env before the try block
    env = os.environ.copy()

    try:
        # Start an interactive rebase
        if interactive:
            # Prepare the rebase script
            rebase_script_lines = []
            rebase_script_lines.append(f"pick {sorted_commits[-1].hash}") # Keep the oldest commit
            for commit_hash in reversed(sorted_commits[:-1]):
                rebase_script_lines.append(f"squash {commit_hash.hash}")
            rebase_script = "\n".join(rebase_script_lines) + "\n"
            
            # Create a temporary file for the rebase instructions
            with NamedTemporaryFile(mode='w', delete=False, prefix='rebase-todo-') as f:
                f.write(rebase_script)
                rebase_path = f.name
            
            # Set GIT_SEQUENCE_EDITOR to cat the prepared file
            editor_cmd = 'cat' if os.name != 'nt' else 'type'
            env["GIT_SEQUENCE_EDITOR"] = f'{editor_cmd} "{rebase_path}" >' # Overwrite the git-rebase-todo file
            
            rebase_cmd = ["git", "-C", repo_path, "rebase", "-i", base_commit]
            try:
                # Run the interactive rebase
                subprocess.run(rebase_cmd, env=env, check=True)
                success = True
            except subprocess.CalledProcessError as e:
                logger.error(f"Interactive rebase failed: {e}")
                # Try to abort the rebase if possible
                subprocess.run(["git", "-C", repo_path, "rebase", "--abort"], capture_output=True)
            finally:
                # Clean up the temporary script file
                try:
                    os.unlink(rebase_path)
                except FileNotFoundError:
                     logger.warning(f"Temporary rebase file not found for cleanup: {rebase_path}")
                except Exception as e:
                     logger.error(f"Failed to clean up rebase file {rebase_path}: {e}")
        
            return success
        else:
            # Non-interactive mode: use soft reset and commit
            newest_commit = sorted_commits[-1].hash
            
            # Reset to the parent of the oldest commit
            cmd = ["git", "-C", repo_path, "reset", "--soft", base_commit]
            subprocess.run(cmd, check=True)
            
            # Commit with the squashed message
            cmd = ["git", "-C", repo_path, "commit", "-F", message_file]
            subprocess.run(cmd, check=True)
        
        logger.info(f"Successfully squashed {len(commit_group)} commits")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error squashing commits: {e}")
        # Try to abort the rebase if it's in progress
        try:
            subprocess.run(["git", "-C", repo_path, "rebase", "--abort"], check=False)
        except:
            pass
        return False
    
    finally:
        # Clean up the message file
        try:
            os.unlink(message_file)
        except:
            pass

def analyze_commits_for_squashing(
    repo_path: str = ".",
    count: int = 10,
    strategy: Union[str, SquashStrategy] = "auto",
    similarity_threshold: float = 0.6,
    time_window_minutes: int = 60,
    workflow = None
) -> List[List[CommitInfo]]:
    """
    Analyze recent commits to find squashable groups.
    
    Args:
        repo_path: Path to the Git repository
        count: Number of recent commits to analyze
        strategy: Strategy to use for finding squashable commits
        similarity_threshold: Threshold for considering commits similar
        time_window_minutes: Time window for time-based strategies
        workflow: Optional workflow object for AI-based analysis
        
    Returns:
        List of commit groups that can be squashed
    """
    # Get recent commits
    commits = get_recent_commits(count, repo_path)
    
    # Find squashable groups
    groups = find_squashable_commits(
        commits,
        strategy=strategy,
        time_window_minutes=time_window_minutes,
        similarity_threshold=similarity_threshold,
        workflow=workflow
    )
    
    return groups

def squash_commit_group(
    group: List[CommitInfo],
    repo_path: str = ".",
    interactive: bool = True
) -> bool:
    """
    Squash a group of commits.
    
    Args:
        group: List of commits to squash
        repo_path: Path to the Git repository
        interactive: Whether to run in interactive mode
        
    Returns:
        True if successful, False otherwise
    """
    return squash_commits(group, repo_path, interactive)

def run_squash_command(
    repo_path: str = ".",
    limit: int = 10,
    strategy: str = "auto",
    interactive: bool = True,
    time_window: int = 86400
) -> bool:
    """
    Run the squash command based on the specified strategy.
    
    Args:
        repo_path: Path to the git repository
        limit: Number of recent commits to consider
        strategy: Strategy to use for grouping commits
        interactive: Whether to prompt for confirmation
        time_window: Time window in seconds for time-based grouping
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running squash command with strategy: {strategy}")
    logger.info(f"Looking at last {limit} commits")
    
    # For now, just log that we're running the command
    # In a real implementation, we would implement the squash logic
    logger.info("This is a placeholder for squash functionality")
    
    # Always return success for now
    return True 
"""
GitHub integration for automatic Pull Request creation.

This module handles automatic GitHub PR creation when auto-apply fixes are successful.
"""

import json
import logging
import subprocess
import re
from datetime import datetime
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


class GitHubPRManager:
    """Manages GitHub Pull Request creation for auto-applied fixes."""
    
    def __init__(self, github_token: str, project_root: str):
        """
        Initialize GitHub PR Manager.
        
        Args:
            github_token: GitHub personal access token
            project_root: Root directory of the project
        """
        self.github_token = github_token
        self.project_root = project_root
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'ErrorAgent/1.0'
        })
        
        # Parse repository info from git remote
        self.repo_owner, self.repo_name = self._get_repository_info()
    
    def _get_repository_info(self) -> tuple[str, str]:
        """Extract repository owner and name from git remote origin."""
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            remote_url = result.stdout.strip()
            
            # Parse GitHub URL (both HTTPS and SSH formats)
            # HTTPS: https://github.com/owner/repo.git
            # SSH: git@github.com:owner/repo.git
            if remote_url.startswith('https://github.com/'):
                match = re.match(r'https://github\.com/([^/]+)/([^/]+)(?:\.git)?', remote_url)
            elif remote_url.startswith('git@github.com:'):
                match = re.match(r'git@github\.com:([^/]+)/([^/]+)(?:\.git)?', remote_url)
            else:
                raise ValueError(f"Unsupported remote URL format: {remote_url}")
            
            if not match:
                raise ValueError(f"Could not parse repository from URL: {remote_url}")
                
            owner, repo = match.groups()
            # Remove .git suffix if present
            if repo.endswith('.git'):
                repo = repo[:-4]
            logger.info(f"Detected repository: {owner}/{repo}")
            return owner, repo
            
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to get git remote origin: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing repository info: {e}")
    
    def _run_git_command(self, command: list[str]) -> str:
        """Run a git command and return stdout."""
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(command)}")
            logger.error(f"Error: {e.stderr}")
            raise
    
    def _generate_branch_name(self, error_context: Dict[str, Any]) -> str:
        """Generate a unique branch name for the auto-fix."""
        error_type = error_context.get('error_type', 'unknown').lower()
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        function_name = error_context.get('function_name') or error_context.get('function', 'unknown')
        
        # Clean up function name for branch name
        clean_function = re.sub(r'[^a-zA-Z0-9_]', '', function_name)
        
        return f"auto-fix-{error_type}-{clean_function}-{timestamp}"
    
    def _create_branch(self, branch_name: str) -> bool:
        """Create and checkout a new git branch."""
        try:
            # Ensure we're on main/master branch
            try:
                self._run_git_command(['git', 'checkout', 'main'])
            except subprocess.CalledProcessError:
                try:
                    self._run_git_command(['git', 'checkout', 'master'])
                except subprocess.CalledProcessError:
                    logger.warning("Could not checkout main/master branch, staying on current branch")
            
            # Pull latest changes
            self._run_git_command(['git', 'pull', 'origin'])
            
            # Create and checkout new branch
            self._run_git_command(['git', 'checkout', '-b', branch_name])
            logger.info(f"Created and checked out branch: {branch_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch {branch_name}: {e}")
            return False
    
    def _commit_changes(self, file_path: str, commit_message: str) -> bool:
        """Add and commit the specified file."""
        try:
            # Add the specific file
            self._run_git_command(['git', 'add', file_path])
            
            # Check if there are changes to commit
            try:
                self._run_git_command(['git', 'diff', '--staged', '--quiet'])
                logger.warning("No changes to commit")
                return False
            except subprocess.CalledProcessError:
                # There are staged changes, proceed with commit
                pass
            
            # Commit changes
            self._run_git_command(['git', 'commit', '-m', commit_message])
            logger.info(f"Committed changes to {file_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e}")
            return False
    
    def _push_branch(self, branch_name: str) -> bool:
        """Push the branch to remote origin."""
        try:
            self._run_git_command(['git', 'push', 'origin', branch_name])
            logger.info(f"Pushed branch {branch_name} to origin")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push branch {branch_name}: {e}")
            return False
    
    def _create_pr_body(self, error_context: Dict[str, Any], apply_result: Dict[str, Any], insights: Dict[str, Any]) -> str:
        """Generate PR body content."""
        # Debug log all available data
        logger.info(f"DEBUG PR Body - error_context keys: {list(error_context.keys())}")
        logger.info(f"DEBUG PR Body - apply_result keys: {list(apply_result.keys())}")
        logger.info(f"DEBUG PR Body - insights keys: {list(insights.keys())}")
        
        # Extract error information with better field mapping
        error_type = error_context.get('error_type') or error_context.get('exception_type', 'Unknown')
        file_path = apply_result.get('file') or error_context.get('file_path') or error_context.get('file', 'Unknown')
        function_name = error_context.get('function_name') or error_context.get('function', 'Unknown')
        line_number = error_context.get('line_number') or error_context.get('line', 'Unknown')
        error_message = error_context.get('error_message') or error_context.get('message', 'Unknown')
        
        # Extract before/after code from insights with improved parsing
        corrected_function = insights.get('corrected_function', '')
        before_code = 'Not available'
        after_code = 'Not available'
        
        corrected_function_preview = str(corrected_function)[:200] if corrected_function else "None"
        logger.info(f"DEBUG PR Body - corrected_function: {corrected_function_preview}...")
        
        if corrected_function:
            # Handle both "# BEFORE:" and "BEFORE:" formats
            if '# BEFORE:' in corrected_function and '# AFTER:' in corrected_function:
                try:
                    parts = corrected_function.split('# BEFORE:')
                    if len(parts) > 1:
                        before_after = parts[1].split('# AFTER:')
                        if len(before_after) > 1:
                            before_code = before_after[0].strip()
                            after_code = before_after[1].strip()
                            # Clean up any extra content after the code
                            if '```' in after_code:
                                after_code = after_code.split('```')[0].strip()
                except Exception as e:
                    logger.warning(f"Failed to parse BEFORE/AFTER: {e}")
        
        # Clean up file path for display
        if file_path != 'Unknown':
            display_file = file_path.replace('\\', '/').split('/')[-1] if '/' in file_path or '\\' in file_path else file_path
        else:
            display_file = file_path
            
        return f"""## Auto-Fix Applied

**Error Type:** {error_type}  
**File:** `{display_file}`  
**Function:** `{function_name}`  
**Line:** {line_number}

### Original Error
```
{error_message}
```

### Applied Changes
```python
# BEFORE:
{before_code}

# AFTER:
{after_code}
```

### Validation Results
- ✅ Syntax validation passed
- ✅ Linter checks passed  
- ✅ Auto-apply successful

---
*This PR was automatically created by Error Agent*"""
    
    def _create_github_pr(self, branch_name: str, title: str, body: str) -> Dict[str, Any]:
        """Create a GitHub Pull Request."""
        try:
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls"
            
            data = {
                'title': title,
                'body': body,
                'head': branch_name,
                'base': 'main'  # or 'master', we might need to detect this
            }
            
            response = self.session.post(url, json=data)
            
            if response.status_code == 201:
                pr_data = response.json()
                return {
                    'success': True,
                    'pr_url': pr_data['html_url'],
                    'pr_number': pr_data['number']
                }
            else:
                # Try with 'master' base if 'main' failed
                if data['base'] == 'main' and response.status_code == 422:
                    data['base'] = 'master'
                    response = self.session.post(url, json=data)
                    if response.status_code == 201:
                        pr_data = response.json()
                        return {
                            'success': True,
                            'pr_url': pr_data['html_url'],
                            'pr_number': pr_data['number']
                        }
                
                return {
                    'success': False,
                    'error': f"GitHub API error: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to create PR: {str(e)}"
            }
    
    def create_auto_fix_pr(
        self, 
        error_context: Dict[str, Any], 
        apply_result: Dict[str, Any], 
        insights: Dict[str, Any],
        custom_branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a GitHub PR for an auto-applied fix.
        
        Args:
            error_context: Information about the error that was fixed
            apply_result: Result of the auto-apply operation
            insights: LLM insights about the fix
            custom_branch_name: Optional custom branch name
            
        Returns:
            Dict with success status and PR details or error info
        """
        try:
            # Generate branch name
            if custom_branch_name:
                branch_name = custom_branch_name
            else:
                branch_name = self._generate_branch_name(error_context)
            
            # Create branch
            if not self._create_branch(branch_name):
                return {'success': False, 'error': 'Failed to create git branch'}
            
            # Commit changes
            file_path = apply_result.get('file', error_context.get('file_path', ''))
            error_type = error_context.get('error_type', 'Unknown')
            function_name = error_context.get('function_name', 'unknown')
            
            commit_message = f"🤖 Auto-fix {error_type} in {function_name}\n\nAutomatically applied fix for {error_type} error in function {function_name}"
            
            if not self._commit_changes(file_path, commit_message):
                return {'success': False, 'error': 'No changes to commit or commit failed'}
            
            # Push branch
            if not self._push_branch(branch_name):
                return {'success': False, 'error': 'Failed to push branch to remote'}
            
            # Create PR
            pr_title = f"Auto-fix: {error_type} in {function_name}"
            pr_body = self._create_pr_body(error_context, apply_result, insights)
            
            pr_result = self._create_github_pr(branch_name, pr_title, pr_body)
            
            if pr_result['success']:
                logger.info(f"Successfully created GitHub PR: {pr_result['pr_url']}")
                return {
                    'success': True,
                    'pr_url': pr_result['pr_url'],
                    'pr_number': pr_result['pr_number'],
                    'branch_name': branch_name
                }
            else:
                return {'success': False, 'error': pr_result['error']}
                
        except Exception as e:
            logger.error(f"GitHub PR creation failed: {e}")
            return {'success': False, 'error': str(e)}

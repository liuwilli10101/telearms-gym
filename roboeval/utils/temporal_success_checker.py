"""
Temporal Success Checker for RoboEval

Provides a modular system for tracking hierarchical, sequential sub-goals
to ensure tasks are completed in the correct temporal order.
"""

from __future__ import annotations
from typing import List, Dict, Callable, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np


class SubGoalStatus(Enum):
    """Status of a sub-goal in the temporal sequence."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # For optional goals


@dataclass
class SubGoal:
    """Represents a single sub-goal in the temporal sequence.
    
    Attributes:
        name: Human-readable name of the sub-goal
        check_fn: Callable that returns True if the sub-goal is satisfied
        description: Optional detailed description
        is_optional: If True, this sub-goal can be skipped without failing the task
        timeout_steps: Maximum steps allowed for this sub-goal (None = no limit)
        metadata: Additional metadata for tracking/debugging
    """
    name: str
    check_fn: Callable[[], bool]
    description: str = ""
    is_optional: bool = False
    timeout_steps: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Internal tracking
    status: SubGoalStatus = SubGoalStatus.NOT_STARTED
    start_step: Optional[int] = None
    completion_step: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    was_false_at_start: bool = False  # Track if goal was false when it became active


class TemporalSuccessChecker:
    """
    Modular temporal success checking system for sequential task goals.
    
    This system enables easy tracking of sub-stage goals and ensures that
    the overall task meets success criteria temporally (in the correct sequence).
    
    Example:
        >>> checker = TemporalSuccessChecker()
        >>> # Define sub-goals in order
        >>> checker.add_sub_goal("pick_blue_cube", lambda: self.is_holding(blue_cube))
        >>> checker.add_sub_goal("pick_red_cube", lambda: self.is_holding(red_cube))
        >>> checker.add_sub_goal("stack_cubes", lambda: self.is_stacked(red_cube, blue_cube))
        >>> 
        >>> # In your _success() method:
        >>> checker.update()
        >>> if checker.is_complete():
        >>>     return True
    """
    
    def __init__(
        self,
        allow_skip_optional: bool = True,
        reset_on_failure: bool = True
    ):
        """
        Initialize the temporal success checker.
        
        Args:
            allow_skip_optional: If True, optional sub-goals can be skipped
            reset_on_failure: If True, failing a sub-goal resets all progress
        """
        self.allow_skip_optional = allow_skip_optional
        self.reset_on_failure = reset_on_failure
        
        self._sub_goals: List[SubGoal] = []
        self._current_index: int = 0
        self._step_count: int = 0
        self._episode_start_time: Optional[float] = None
        self._is_complete: bool = False
        self._has_failed: bool = False
        
        # Metrics
        self._metrics: Dict[str, Any] = {
            "total_steps": 0,
            "total_time": 0.0,
            "sub_goal_completion_times": [],
            "sub_goal_completion_steps": [],
            "failed_sub_goals": [],
            "skipped_sub_goals": [],
        }
    
    def add_sub_goal(
        self,
        name: str,
        check_fn: Callable[[], bool],
        description: str = "",
        is_optional: bool = False,
        timeout_steps: Optional[int] = None,
        **metadata
    ) -> None:
        """
        Add a sub-goal to the temporal sequence.
        
        Args:
            name: Unique identifier for the sub-goal
            check_fn: Function that returns True when sub-goal is satisfied
            description: Human-readable description
            is_optional: Whether this sub-goal can be skipped
            timeout_steps: Maximum steps allowed for completion
            **metadata: Additional tracking information
        """
        sub_goal = SubGoal(
            name=name,
            check_fn=check_fn,
            description=description,
            is_optional=is_optional,
            timeout_steps=timeout_steps,
            metadata=metadata
        )
        self._sub_goals.append(sub_goal)
    
    def reset(self) -> None:
        """Reset the checker to initial state."""
        self._current_index = 0
        self._step_count = 0
        self._episode_start_time = time.time()
        self._is_complete = False
        self._has_failed = False
        
        # Reset all sub-goals
        for goal in self._sub_goals:
            goal.status = SubGoalStatus.NOT_STARTED
            goal.start_step = None
            goal.completion_step = None
            goal.start_time = None
            goal.completion_time = None
        
        # Reset metrics
        self._metrics = {
            "total_steps": 0,
            "total_time": 0.0,
            "sub_goal_completion_times": [],
            "sub_goal_completion_steps": [],
            "failed_sub_goals": [],
            "skipped_sub_goals": [],
        }
    
    def update(self) -> bool:
        """
        Update the temporal checker state (call every step).
        
        Returns:
            True if all sub-goals completed successfully, False otherwise
        """
        if self._is_complete or self._has_failed:
            return self._is_complete
        
        if self._episode_start_time is None:
            self._episode_start_time = time.time()
        
        self._step_count += 1
        
        # Always use strict sequential ordering
        return self._update_strict_order()
    
    def _update_strict_order(self) -> bool:
        """Update logic for strict sequential ordering."""
        # Check if we've completed all goals
        if self._current_index >= len(self._sub_goals):
            self._is_complete = True
            self._finalize_metrics()
            return True
        
        current_goal = self._sub_goals[self._current_index]
        
        # Mark as in progress if just started
        if current_goal.status == SubGoalStatus.NOT_STARTED:
            current_goal.status = SubGoalStatus.IN_PROGRESS
            current_goal.start_step = self._step_count
            current_goal.start_time = time.time()
            
            # Check if goal is already satisfied when it becomes active
            try:
                is_satisfied = current_goal.check_fn()
                current_goal.was_false_at_start = not is_satisfied
                
                if is_satisfied:
                    print(f"⚠️ Warning: Sub-goal '{current_goal.name}' was ALREADY satisfied when it became active!")
                    print(f"   This violates temporal ordering - goal must transition from false→true AFTER becoming active.")
                    print(f"   Goal will NOT complete until it becomes false, then true again.")
            except Exception as e:
                print(f"⚠️ Error checking initial state of sub-goal '{current_goal.name}': {e}")
                current_goal.was_false_at_start = True  # Assume false on error
        
        # Check for timeout
        if current_goal.timeout_steps is not None:
            elapsed_steps = self._step_count - current_goal.start_step
            if elapsed_steps > current_goal.timeout_steps:
                if current_goal.is_optional and self.allow_skip_optional:
                    self._skip_current_goal()
                    return self.update()  # Check next goal
                else:
                    self._fail_current_goal("Timeout")
                    return False
        
        # Check if current goal is satisfied
        try:
            is_satisfied = current_goal.check_fn()
            
            # Goal can only complete if it was false when it started
            # OR if it has transitioned from true → false → true
            if is_satisfied:
                if current_goal.was_false_at_start:
                    # Goal was false at start and is now true - valid completion
                    self._complete_current_goal()
                    # Automatically check next goal in same step
                    return self.update()
                else:
                    # Goal is still satisfied but was already true at start
                    # We need to wait for it to become false first
                    pass
            else:
                # Goal became false - now it can complete when it becomes true again
                if not current_goal.was_false_at_start:
                    current_goal.was_false_at_start = True
                    print(f"✓ Sub-goal '{current_goal.name}' is now false - can complete when it becomes true")
                    
        except Exception as e:
            print(f"⚠️ Error checking sub-goal '{current_goal.name}': {e}")
            if not current_goal.is_optional:
                self._fail_current_goal(f"Exception: {e}")
                return False
        
        return False
    
    def _complete_current_goal(self) -> None:
        """Mark the current sub-goal as completed."""
        current_goal = self._sub_goals[self._current_index]
        current_goal.status = SubGoalStatus.COMPLETED
        current_goal.completion_step = self._step_count
        current_goal.completion_time = time.time()
        
        # Record metrics
        elapsed_steps = current_goal.completion_step - current_goal.start_step
        elapsed_time = current_goal.completion_time - current_goal.start_time
        
        self._metrics["sub_goal_completion_steps"].append({
            "name": current_goal.name,
            "steps": elapsed_steps
        })
        self._metrics["sub_goal_completion_times"].append({
            "name": current_goal.name,
            "time": elapsed_time
        })
        
        print(f"✅ Sub-goal '{current_goal.name}' completed "
              f"(step {current_goal.completion_step}, {elapsed_time:.2f}s)")
        
        self._current_index += 1
    
    def _skip_current_goal(self) -> None:
        """Mark the current optional sub-goal as skipped."""
        current_goal = self._sub_goals[self._current_index]
        current_goal.status = SubGoalStatus.SKIPPED
        
        self._metrics["skipped_sub_goals"].append(current_goal.name)
        
        print(f"⏭️ Sub-goal '{current_goal.name}' skipped (optional)")
        
        self._current_index += 1
    
    def _fail_current_goal(self, reason: str = "") -> None:
        """Mark the current sub-goal as failed."""
        current_goal = self._sub_goals[self._current_index]
        current_goal.status = SubGoalStatus.FAILED
        
        self._metrics["failed_sub_goals"].append({
            "name": current_goal.name,
            "reason": reason,
            "step": self._step_count
        })
        
        print(f"❌ Sub-goal '{current_goal.name}' failed: {reason}")
        
        if self.reset_on_failure:
            self.reset()
        else:
            self._has_failed = True
    
    def _finalize_metrics(self) -> None:
        """Finalize metrics when all goals are complete."""
        self._metrics["total_steps"] = self._step_count
        if self._episode_start_time:
            self._metrics["total_time"] = time.time() - self._episode_start_time
    
    def is_complete(self) -> bool:
        """Check if all sub-goals have been completed successfully."""
        return self._is_complete
    
    def has_failed(self) -> bool:
        """Check if any sub-goal has failed."""
        return self._has_failed
    
    def get_current_sub_goal(self) -> Optional[SubGoal]:
        """Get the currently active sub-goal."""
        if 0 <= self._current_index < len(self._sub_goals):
            return self._sub_goals[self._current_index]
        return None
    
    def get_progress(self) -> Tuple[int, int]:
        """
        Get progress as (completed_count, total_count).
        
        Returns:
            Tuple of (number of completed goals, total number of goals)
        """
        completed = sum(1 for g in self._sub_goals if g.status == SubGoalStatus.COMPLETED)
        return completed, len(self._sub_goals)
    
    def get_progress_percentage(self) -> float:
        """Get progress as a percentage (0.0 to 100.0)."""
        if not self._sub_goals:
            return 100.0
        completed, total = self.get_progress()
        return (completed / total) * 100.0
    
    def get_sub_goal_status(self, name: str) -> Optional[SubGoalStatus]:
        """Get the status of a specific sub-goal by name."""
        for goal in self._sub_goals:
            if goal.name == name:
                return goal.status
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics about the temporal execution."""
        return self._metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        completed, total = self.get_progress()
        current_goal = self.get_current_sub_goal()
        
        return {
            "is_complete": self._is_complete,
            "has_failed": self._has_failed,
            "progress": f"{completed}/{total}",
            "progress_percentage": self.get_progress_percentage(),
            "current_sub_goal": current_goal.name if current_goal else None,
            "total_steps": self._step_count,
            "sub_goals": [
                {
                    "name": g.name,
                    "status": g.status.value,
                    "is_optional": g.is_optional,
                    "completion_step": g.completion_step
                }
                for g in self._sub_goals
            ]
        }
    
    def print_status(self) -> None:
        """Print a human-readable status summary."""
        print("\n" + "="*60)
        print("📊 Temporal Success Checker Status")
        print("="*60)
        
        completed, total = self.get_progress()
        print(f"Progress: {completed}/{total} ({self.get_progress_percentage():.1f}%)")
        print(f"Steps: {self._step_count}")
        
        if self._is_complete:
            print("✅ Status: COMPLETE")
        elif self._has_failed:
            print("❌ Status: FAILED")
        else:
            print("🔄 Status: IN PROGRESS")
        
        print("\nSub-Goals:")
        for i, goal in enumerate(self._sub_goals):
            status_icon = {
                SubGoalStatus.NOT_STARTED: "⬜",
                SubGoalStatus.IN_PROGRESS: "🔄",
                SubGoalStatus.COMPLETED: "✅",
                SubGoalStatus.FAILED: "❌",
                SubGoalStatus.SKIPPED: "⏭️",
            }[goal.status]
            
            current_marker = " 👉" if i == self._current_index else ""
            optional_marker = " (optional)" if goal.is_optional else ""
            
            print(f"  {status_icon} {goal.name}{optional_marker}{current_marker}")
            
            if goal.completion_step is not None:
                print(f"     Completed at step {goal.completion_step}")
        
        print("="*60 + "\n")

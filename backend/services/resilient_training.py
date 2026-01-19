"""
Resilient Model Training Service

Provides checkpointing, graceful shutdown, and resume capabilities for model training.
Can run training in background thread with progress tracking.
"""

import os
import sys
import json
import signal
import threading
import subprocess
import time
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from utils.logger import logger

# Platform detection for signal handling
IS_WINDOWS = platform.system() == 'Windows'

class ResilientTrainingService:
    """Service for managing resilient model training with checkpointing"""
    
    def __init__(self, models_dir: str = None):
        from config import Config
        self.models_dir = models_dir or Config.MODELS_DIR
        self.checkpoint_dir = os.path.join(self.models_dir, '.training_checkpoints')
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'training_state.json')
        self.lock_file = os.path.join(self.checkpoint_dir, 'training.lock')
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self._training_process: Optional[subprocess.Popen] = None
        self._training_thread: Optional[threading.Thread] = None
        self._is_training = False
        self._training_lock = threading.Lock()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, gracefully shutting down training...")
            self.stop_training(graceful=True)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def is_training(self) -> bool:
        """Check if training is currently running"""
        with self._training_lock:
            return self._is_training
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        status = {
            'is_training': self.is_training(),
            'has_checkpoint': os.path.exists(self.checkpoint_file),
            'lock_exists': os.path.exists(self.lock_file),
        }
        
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    status['checkpoint'] = checkpoint
            except Exception as e:
                logger.warning(f"Could not read checkpoint: {e}")
        
        if os.path.exists(self.lock_file):
            try:
                with open(self.lock_file, 'r') as f:
                    lock_info = json.load(f)
                    status['lock_info'] = lock_info
            except Exception as e:
                logger.warning(f"Could not read lock file: {e}")
        
        return status
    
    def save_checkpoint(self, step: str, progress: Dict[str, Any]):
        """Save training checkpoint"""
        try:
            checkpoint = {
                'step': step,
                'progress': progress,
                'timestamp': datetime.now().isoformat(),
                'pid': os.getpid(),
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.debug(f"Saved checkpoint: {step}")
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")
    
    def clear_checkpoint(self):
        """Clear training checkpoint"""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except Exception as e:
            logger.warning(f"Could not clear checkpoint: {e}")
    
    def acquire_lock(self) -> bool:
        """Acquire training lock to prevent concurrent training"""
        if os.path.exists(self.lock_file):
            # Check if process is still running
            try:
                with open(self.lock_file, 'r') as f:
                    lock_info = json.load(f)
                    pid = lock_info.get('pid')
                    if pid and self._is_process_running(pid):
                        logger.warning(f"Training lock held by process {pid}")
                        return False
            except Exception:
                pass
            
            # Lock is stale, remove it
            try:
                os.remove(self.lock_file)
            except Exception:
                pass
        
        # Create new lock
        try:
            lock_info = {
                'pid': os.getpid(),
                'timestamp': datetime.now().isoformat(),
            }
            with open(self.lock_file, 'w') as f:
                json.dump(lock_info, f)
            return True
        except Exception as e:
            logger.error(f"Could not acquire lock: {e}")
            return False
    
    def release_lock(self):
        """Release training lock"""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except Exception as e:
            logger.warning(f"Could not release lock: {e}")
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if process is still running"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def start_training(self, background: bool = True, force: bool = False) -> Dict[str, Any]:
        """
        Start model training
        
        Args:
            background: If True, run in background thread
            force: If True, force start even if already training
            
        Returns:
            Status dict with training info
        """
        with self._training_lock:
            if self._is_training and not force:
                return {
                    'success': False,
                    'error': 'Training already in progress',
                    'status': self.get_training_status()
                }
            
            if not self.acquire_lock():
                return {
                    'success': False,
                    'error': 'Could not acquire training lock (another process may be training)',
                    'status': self.get_training_status()
                }
            
            self._is_training = True
        
        if background:
            # Start in background thread
            self._training_thread = threading.Thread(
                target=self._run_training_async,
                name="ResilientTraining",
                daemon=False  # Non-daemon so it can complete
            )
            self._training_thread.start()
            
            return {
                'success': True,
                'message': 'Training started in background',
                'thread_name': self._training_thread.name
            }
        else:
            # Run synchronously
            try:
                return self._run_training_sync()
            finally:
                with self._training_lock:
                    self._is_training = False
                    self.release_lock()
    
    def _run_training_sync(self) -> Dict[str, Any]:
        """Run training synchronously"""
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'scripts',
            'retrain_models_simple.py'
        )
        
        if not os.path.exists(script_path):
            return {
                'success': False,
                'error': f'Training script not found: {script_path}'
            }
        
        try:
            logger.info(f"Starting training script: {script_path}")
            
            # Run training with output streaming
            # Create process group on Unix for better signal handling
            kwargs = {
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
                'text': True,
                'bufsize': 1,
                'universal_newlines': True,
            }
            if not IS_WINDOWS:
                kwargs['preexec_fn'] = os.setsid  # Create new process group for signal handling
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                **kwargs
            )
            
            self._training_process = process
            
            # Stream output
            for line in process.stdout:
                logger.info(f"[TRAINING] {line.rstrip()}")
            
            process.wait()
            
            if process.returncode == 0:
                self.clear_checkpoint()
                return {
                    'success': True,
                    'message': 'Training completed successfully',
                    'returncode': process.returncode
                }
            else:
                return {
                    'success': False,
                    'error': f'Training failed with exit code {process.returncode}',
                    'returncode': process.returncode
                }
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self._training_process = None
    
    def _run_training_async(self):
        """Run training asynchronously in background thread"""
        try:
            result = self._run_training_sync()
            if result.get('success'):
                logger.info("✅ Background training completed successfully!")
            else:
                logger.error(f"❌ Background training failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"❌ Background training error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            with self._training_lock:
                self._is_training = False
                self.release_lock()
                self._training_thread = None
    
    def stop_training(self, graceful: bool = True) -> Dict[str, Any]:
        """
        Stop training
        
        Args:
            graceful: If True, wait for current step to complete
            
        Returns:
            Status dict
        """
        with self._training_lock:
            if not self._is_training:
                return {
                    'success': False,
                    'error': 'No training in progress'
                }
            
            if self._training_process:
                try:
                    if IS_WINDOWS:
                        # Windows signal handling
                        if graceful:
                            self._training_process.terminate()
                            try:
                                self._training_process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                self._training_process.kill()
                                self._training_process.wait()
                        else:
                            self._training_process.kill()
                            self._training_process.wait()
                    else:
                        # Unix signal handling with process groups
                        if graceful:
                            # Send SIGTERM to process group
                            os.killpg(os.getpgid(self._training_process.pid), signal.SIGTERM)
                            # Wait up to 30 seconds
                            try:
                                self._training_process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                # Force kill
                                os.killpg(os.getpgid(self._training_process.pid), signal.SIGKILL)
                                self._training_process.wait()
                        else:
                            # Force kill immediately
                            os.killpg(os.getpgid(self._training_process.pid), signal.SIGKILL)
                            self._training_process.wait()
                    
                    logger.info("Training process stopped")
                except Exception as e:
                    logger.error(f"Error stopping training process: {e}")
                
                self._training_process = None
            
            self._is_training = False
            self.release_lock()
            
            return {
                'success': True,
                'message': 'Training stopped'
            }


# Singleton instance
_training_service: Optional[ResilientTrainingService] = None

def get_resilient_training_service() -> ResilientTrainingService:
    """Get singleton instance of resilient training service"""
    global _training_service
    if _training_service is None:
        _training_service = ResilientTrainingService()
    return _training_service

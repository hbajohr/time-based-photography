#!/usr/bin/env python3
"""
================================================================================
TIME-BASED PHOTOGRAPHY - APPLE SILICON OPTIMIZED
================================================================================

Original concept and implementation by Hannes Bajohr (2015-2022)
Performance optimizations for Apple Silicon (2024)

DESCRIPTION:
    This script creates "time-based photographs" from video files. Instead of
    viewing a video normally (where each frame shows a moment in time), this
    tool extracts a thin vertical "slice" from each frame and concatenates
    them horizontally into a panoramic image. The result compresses an entire
    video's timeline into a single static image.
    
    The script can also generate multiple panoramas (one for each possible
    slice position) and combine them into a new video, creating a "derivative"
    that shows how perspective shifts across the frame.

INSTALLATION:
    1. Ensure Python 3.8+ is installed
    2. Install required dependencies:
       
       pip install numpy pillow tqdm av
       
    3. Install optional dependencies for better performance:
       
       pip install mlx                           # GPU acceleration (Apple Silicon)
       pip install rife-ncnn-vulkan-python-tntwise  # High-quality frame interpolation (Apple Silicon)
    
    4. Ensure FFmpeg is installed (for some interpolation features):
       
       brew install ffmpeg

BASIC USAGE:
    # Simple panorama generation
    python tbp_optimized.py input_video.mp4 output_folder/
    
    # With dimension swap (if output looks scrambled)
    python tbp_optimized.py input_video.mp4 output_folder/ --swap-dimensions
    
    # Create output video from panoramas
    python tbp_optimized.py input_video.mp4 output_folder/ --make-video
    
    # Frame interpolation for low-fps videos
    python tbp_optimized.py input_video.mp4 output_folder/ --interpolate 4

DEPENDENCIES:
    Required:
        - numpy          : Array operations
        - Pillow (PIL)   : Image I/O
        - tqdm           : Progress bars
        - av (PyAV)      : Video decoding with hardware acceleration
        
    Optional:
        - mlx                          : Apple Silicon GPU acceleration for array ops
        - rife-ncnn-vulkan-python-tntwise : Neural network frame interpolation (Apple Silicon)
        - opencv-python                : Fallback video decoding and optical flow

APPLE SILICON OPTIMIZATIONS:
    1. VideoToolbox hardware video decoding via PyAV
    2. MLX framework for GPU-accelerated array operations
    3. ThreadPoolExecutor (unified memory = no copy overhead)
    4. Hardware video encoding for output

LICENSE:
    MIT License - See original project for details

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np                          # Core array operations
from PIL import Image                       # Image loading and saving
import os                                   # File system operations
import sys                                  # System-specific parameters
import time                                 # Timing and benchmarks
import gc                                   # Garbage collection for memory management
import subprocess                           # For FFmpeg interpolation
import tempfile                             # Temporary file handling
import shutil                               # File operations (cleanup)
from pathlib import Path                    # Modern path handling
from concurrent.futures import ThreadPoolExecutor, as_completed  # Parallel processing
from threading import Lock                  # Thread safety for file writes
from typing import Tuple, List, Optional    # Type hints
import argparse                             # Command-line argument parsing

# -----------------------------------------------------------------------------
# Optional dependency: PyAV for hardware-accelerated video decoding
# PyAV wraps FFmpeg and can use Apple's VideoToolbox for H.264/HEVC decoding
# -----------------------------------------------------------------------------
try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    import cv2  # Fallback to OpenCV if PyAV not available

# -----------------------------------------------------------------------------
# Optional dependency: MLX for GPU acceleration
# MLX is Apple's machine learning framework optimized for Apple Silicon
# It allows array operations to run on the GPU
# -----------------------------------------------------------------------------
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# -----------------------------------------------------------------------------
# Optional dependency: RIFE for high-quality frame interpolation
# RIFE is a neural network that generates intermediate frames
# The ncnn-vulkan version runs efficiently on Apple Silicon
# Two packages exist: the original and a fork with Apple Silicon support
# NOTE: RIFE packages can have compatibility issues with NumPy 2.x
# We use lazy loading to avoid import errors when RIFE isn't needed
# -----------------------------------------------------------------------------
HAS_RIFE = False
RIFE_ERROR = None
_rife_module = None

def _check_rife_available():
    """Check if RIFE is available without fully importing it."""
    global HAS_RIFE, RIFE_ERROR
    try:
        import importlib.util
        spec = importlib.util.find_spec("rife_ncnn_vulkan_python")
        if spec is not None:
            HAS_RIFE = True
        else:
            RIFE_ERROR = "Module not found"
    except Exception as e:
        RIFE_ERROR = str(e)

def _get_rife_class():
    """Lazy load RIFE class only when actually needed."""
    global _rife_module, HAS_RIFE, RIFE_ERROR
    if _rife_module is not None:
        return _rife_module
    
    try:
        from rife_ncnn_vulkan_python import Rife
        _rife_module = Rife
        return Rife
    except ImportError as e:
        HAS_RIFE = False
        RIFE_ERROR = f"Import failed: {e}"
        return None
    except Exception as e:
        HAS_RIFE = False
        RIFE_ERROR = f"Error: {e}"
        return None

# Just check if module exists, don't import it yet
_check_rife_available()
    
# Store which RIFE package is available for help messages
RIFE_INSTALL_CMD = "pip install rife-ncnn-vulkan-python-tntwise  # Apple Silicon"

# -----------------------------------------------------------------------------
# Optional dependency: tqdm for progress bars
# Falls back to a passthrough function if not available
# -----------------------------------------------------------------------------
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Dummy tqdm that just returns the iterable unchanged."""
        return iterable


# =============================================================================
# FRAME INTERPOLATION
# =============================================================================

class FrameInterpolator:
    """
    Frame interpolation to increase effective frame rate.
    
    This is useful when the source video has a low frame rate, which would
    result in a narrow output panorama. By interpolating frames, we can
    artificially increase the frame count and thus the panorama width.
    
    Supports three backends:
        - FFmpeg minterpolate: Fast, good quality, no extra dependencies (RECOMMENDED)
        - RIFE: Best quality, uses neural networks, requires rife-ncnn-vulkan-python-tntwise
        - OpenCV: Basic quality using optical flow, no extra dependencies
    
    Attributes:
        method (str): The interpolation method being used
        rife: RIFE model instance (if using RIFE method)
    """
    
    def __init__(self, method: str = 'auto'):
        """
        Initialize the frame interpolator.
        
        Args:
            method: Interpolation method to use. Options:
                    - 'auto': Use FFmpeg (most reliable)
                    - 'ffmpeg': Use FFmpeg's minterpolate filter
                    - 'rife': Use RIFE neural network
                    - 'opencv': Use OpenCV optical flow
        """
        # Auto now defaults to FFmpeg since it's most reliable
        if method == 'auto':
            method = 'ffmpeg'
        
        self.method = method
        self.rife = None
        
        # Handle RIFE initialization with proper error handling
        if method == 'rife':
            if not HAS_RIFE:
                print(f"WARNING: RIFE not available, falling back to FFmpeg")
                if RIFE_ERROR:
                    print(f"         Reason: {RIFE_ERROR}")
                print(f"         Install with: {RIFE_INSTALL_CMD}")
                self.method = 'ffmpeg'
            else:
                # Lazy load RIFE only when actually needed
                Rife = _get_rife_class()
                if Rife is None:
                    print(f"WARNING: RIFE import failed, falling back to FFmpeg")
                    if RIFE_ERROR:
                        print(f"         Reason: {RIFE_ERROR}")
                    self.method = 'ffmpeg'
                else:
                    # Try to initialize RIFE - this can fail if models are missing
                    try:
                        self.rife = Rife(gpuid=0)
                    except FileNotFoundError as e:
                        print(f"WARNING: RIFE model files not found, falling back to FFmpeg")
                        print(f"         Error: {e}")
                        print(f"         You may need to download RIFE models separately")
                        self.method = 'ffmpeg'
                    except Exception as e:
                        print(f"WARNING: RIFE initialization failed, falling back to FFmpeg")
                        print(f"         Error: {e}")
                        self.method = 'ffmpeg'
    
    def interpolate_video_ffmpeg(self, input_path: str, output_path: str, 
                                  multiplier: int = 2, quality: str = 'medium') -> str:
        """
        Interpolate video using FFmpeg's minterpolate filter.
        
        This method creates a new video file with interpolated frames.
        Uses hardware acceleration on Apple Silicon where possible.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output interpolated video
            multiplier: Frame rate multiplier (2 = double fps, 4 = quadruple)
            quality: Interpolation quality preset:
                     - 'fast': Blend mode (~5x faster, lower quality)
                     - 'medium': MCI with OBMC (~2x faster than best, good quality)
                     - 'best': MCI with AOBMC + bidir (slowest, highest quality)
        
        Returns:
            Path to the output video file
        
        Raises:
            RuntimeError: If FFmpeg/FFprobe are not available
        """
        import re
        import platform
        
        # Check if FFprobe is available
        try:
            subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFprobe not found. Please install FFmpeg:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu: sudo apt install ffmpeg"
            )
        
        # Check for Apple Silicon / macOS for hardware acceleration
        is_apple_silicon = (
            platform.system() == 'Darwin' and 
            platform.machine() == 'arm64'
        )
        
        # Check if VideoToolbox encoder is available
        has_videotoolbox = False
        if is_apple_silicon:
            check_cmd = ['ffmpeg', '-hide_banner', '-encoders']
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            has_videotoolbox = 'h264_videotoolbox' in result.stdout
        
        # Get video duration and frame rate
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,nb_frames,duration',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1',
            input_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        probe_output = result.stdout
        
        # Parse frame rate
        input_fps = 30.0  # Default fallback
        duration_secs = 0.0
        total_frames = 0
        
        for line in probe_output.split('\n'):
            line = line.strip()
            if line.startswith('r_frame_rate='):
                fps_str = line.split('=')[1]
                try:
                    if '/' in fps_str:
                        parts = fps_str.split('/')
                        num = int(parts[0].strip())
                        den = int(parts[1].strip())
                        if den != 0:
                            input_fps = num / den
                    else:
                        input_fps = float(fps_str)
                except (ValueError, IndexError):
                    pass
            elif line.startswith('duration='):
                try:
                    duration_secs = float(line.split('=')[1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('nb_frames='):
                try:
                    total_frames = int(line.split('=')[1])
                except (ValueError, IndexError):
                    pass
        
        # Calculate expected output frames
        if total_frames == 0 and duration_secs > 0:
            total_frames = int(duration_secs * input_fps)
        
        target_fps = input_fps * multiplier
        expected_output_frames = total_frames * multiplier
        
        # Quality presets for minterpolate filter
        # - mi_mode: dup (duplicate), blend, mci (motion compensated)
        # - mc_mode: obmc (overlapped block), aobmc (adaptive - slower but better)
        # - me_mode: bidir (bidirectional), bilat (bilateral - faster)
        # - vsbmc: variable size block motion compensation (slower but better)
        quality_presets = {
            'fast': {
                'filter': f"minterpolate='fps={target_fps}:mi_mode=blend'",
                'desc': 'blend (~5x faster)'
            },
            'medium': {
                'filter': f"minterpolate='fps={target_fps}:mi_mode=mci:mc_mode=obmc:me_mode=bilat:vsbmc=0'",
                'desc': 'MCI/OBMC (~2x faster)'
            },
            'best': {
                'filter': f"minterpolate='fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'",
                'desc': 'MCI/AOBMC (highest quality)'
            }
        }
        
        preset = quality_presets.get(quality, quality_presets['medium'])
        filter_str = preset['filter']
        mode_desc = preset['desc']
        
        print(f"Interpolating {input_fps:.1f} fps → {target_fps:.1f} fps")
        print(f"         Quality: {quality} - {mode_desc}")
        print(f"         Input: {total_frames} frames, Output: ~{expected_output_frames} frames")
        
        if has_videotoolbox:
            print(f"         Hardware: VideoToolbox encode (Apple Silicon)")
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Hardware decoding on Apple Silicon
        if is_apple_silicon:
            cmd.extend(['-hwaccel', 'videotoolbox'])
        
        # Input file
        cmd.extend(['-i', input_path])
        
        # Video filter (minterpolate runs on CPU regardless)
        cmd.extend(['-filter:v', filter_str])
        
        # Encoder selection - use hardware encoding if available
        if has_videotoolbox:
            # VideoToolbox H.264 encoder (hardware accelerated)
            cmd.extend([
                '-c:v', 'h264_videotoolbox',
                '-q:v', '65',              # Quality (0-100, higher=better)
                '-allow_sw', '1',          # Allow software fallback
            ])
        else:
            # Software encoding fallback
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
            ])
        
        # Pixel format for compatibility
        cmd.extend(['-pix_fmt', 'yuv420p'])
        
        # Progress output
        cmd.extend([
            '-progress', 'pipe:1',
            '-stats_period', '0.5',
        ])
        
        # Output file
        cmd.append(output_path)
        
        # Run FFmpeg with progress parsing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Create progress bar
        pbar = tqdm(total=expected_output_frames, desc="Interpolating", unit="frames")
        last_frame = 0
        
        # Parse progress output
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                line = line.strip()
                # FFmpeg progress outputs "frame=N" lines
                if line.startswith('frame='):
                    try:
                        current_frame = int(line.split('=')[1])
                        if current_frame > last_frame:
                            pbar.update(current_frame - last_frame)
                            last_frame = current_frame
                    except (ValueError, IndexError):
                        pass
        except KeyboardInterrupt:
            process.terminate()
            pbar.close()
            raise
        
        pbar.close()
        
        # Check if FFmpeg succeeded
        stderr_output = process.stderr.read()
        if process.returncode != 0:
            print(f"WARNING: FFmpeg interpolation may have failed (return code: {process.returncode})")
            if stderr_output:
                # Show last few lines of error
                error_lines = stderr_output.strip().split('\n')[-5:]
                for line in error_lines:
                    print(f"         {line}")
        else:
            print(f"         Interpolation complete!")
        
        return output_path
    
    def interpolate_frames_rife(self, frames: np.ndarray, multiplier: int = 2) -> np.ndarray:
        """
        Interpolate frames using RIFE neural network.
        
        RIFE (Real-time Intermediate Flow Estimation) is a deep learning model
        that generates high-quality intermediate frames. It's particularly good
        at handling complex motion and produces smoother results than traditional
        optical flow methods.
        
        For multipliers > 2, this method uses recursive interpolation:
        - 4x = two passes of 2x interpolation
        - 8x = three passes of 2x interpolation
        
        Args:
            frames: Input frames as numpy array with shape (N, H, W, 3)
            multiplier: Frame count multiplier (2, 4, or 8)
        
        Returns:
            Interpolated frames as numpy array
        """
        if not HAS_RIFE:
            raise RuntimeError(f"RIFE not available - install with: {RIFE_INSTALL_CMD}")
        
        # Calculate number of 2x passes needed for the requested multiplier
        passes = {2: 1, 4: 2, 8: 3}.get(multiplier, 1)
        
        current_frames = frames
        
        # Perform recursive 2x interpolation
        for p in range(passes):
            print(f"RIFE interpolation pass {p+1}/{passes} "
                  f"({len(current_frames)} → {len(current_frames)*2-1} frames)...")
            
            interpolated = []
            
            # Process each pair of adjacent frames
            for i in tqdm(range(len(current_frames) - 1), desc=f"RIFE pass {p+1}"):
                # Convert numpy arrays to PIL Images for RIFE
                frame1 = Image.fromarray(current_frames[i])
                frame2 = Image.fromarray(current_frames[i + 1])
                
                # Add the original frame
                interpolated.append(current_frames[i])
                
                # Generate intermediate frame using RIFE
                # RIFE automatically generates the frame at t=0.5
                mid_frame = self.rife.process(frame1, frame2)
                interpolated.append(np.array(mid_frame))
            
            # Don't forget the last frame
            interpolated.append(current_frames[-1])
            current_frames = np.array(interpolated)
        
        print(f"RIFE interpolation complete: {len(frames)} → {len(current_frames)} frames")
        return current_frames
    
    def interpolate_frames_opencv(self, frames: np.ndarray, multiplier: int = 2) -> np.ndarray:
        """
        Interpolate frames using OpenCV optical flow.
        
        This is a simpler, traditional computer vision approach that:
        1. Calculates optical flow between adjacent frames
        2. Warps the first frame using the flow field to create intermediate frames
        
        Quality is lower than RIFE but requires no additional dependencies.
        
        Args:
            frames: Input frames as numpy array with shape (N, H, W, 3)
            multiplier: Frame count multiplier (2, 4, or 8)
        
        Returns:
            Interpolated frames as numpy array
        """
        import cv2
        
        # Calculate number of 2x passes needed
        passes = {2: 1, 4: 2, 8: 3}.get(multiplier, 1)
        
        current_frames = frames
        
        for p in range(passes):
            print(f"OpenCV interpolation pass {p+1}/{passes} "
                  f"({len(current_frames)} → {len(current_frames)*2-1} frames)...")
            
            interpolated = []
            
            for i in tqdm(range(len(current_frames) - 1), desc=f"Optical flow pass {p+1}"):
                frame1 = current_frames[i]
                frame2 = current_frames[i + 1]
                
                # Add original frame
                interpolated.append(frame1)
                
                # Convert to grayscale for optical flow calculation
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                
                # Calculate dense optical flow using Farneback method
                # This gives us a flow vector (dx, dy) for each pixel
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, 
                    None,              # No initial flow estimate
                    pyr_scale=0.5,     # Pyramid scale (0.5 = classical pyramid)
                    levels=3,          # Number of pyramid levels
                    winsize=15,        # Averaging window size
                    iterations=3,      # Iterations at each level
                    poly_n=5,          # Size of pixel neighborhood
                    poly_sigma=1.2,    # Gaussian std for smoothing
                    flags=0
                )
                
                # Generate intermediate frame at t=0.5
                h, w = frame1.shape[:2]
                
                # Create coordinate grids
                y, x = np.mgrid[0:h, 0:w].astype(np.float32)
                
                # Warp coordinates using half the flow (t=0.5)
                map_x = x + flow[..., 0] * 0.5
                map_y = y + flow[..., 1] * 0.5
                
                # Remap/warp the first frame using the calculated coordinates
                mid_frame = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)
                interpolated.append(mid_frame)
            
            # Add last frame
            interpolated.append(current_frames[-1])
            current_frames = np.array(interpolated)
        
        print(f"OpenCV interpolation complete: {len(frames)} → {len(current_frames)} frames")
        return current_frames


# =============================================================================
# VIDEO READER
# =============================================================================

class VideoReader:
    """
    Hardware-accelerated video reader.
    
    This class abstracts video reading and supports two backends:
        - PyAV with VideoToolbox (hardware-accelerated on Apple Silicon)
        - OpenCV (software decoding fallback)
    
    VideoToolbox is Apple's framework for hardware video encoding/decoding.
    On Apple Silicon, it uses the dedicated media engine for very fast
    H.264/HEVC decoding with minimal CPU usage.
    
    Attributes:
        input_file (str): Path to the video file
        use_hardware (bool): Whether hardware decoding is enabled
        frame_count (int): Total number of frames in the video
        frame_width (int): Width of each frame in pixels
        frame_height (int): Height of each frame in pixels
        fps (float): Frame rate of the video
    """
    
    def __init__(self, input_file: str, use_hardware: bool = True):
        """
        Initialize the video reader.
        
        Args:
            input_file: Path to the video file to read
            use_hardware: If True, attempt to use VideoToolbox hardware decoding
                         via PyAV. Falls back to OpenCV if PyAV not available.
        """
        self.input_file = input_file
        self.use_hardware = use_hardware and HAS_PYAV
        
        if self.use_hardware:
            self._init_pyav()
        else:
            self._init_opencv()
    
    def _init_pyav(self):
        """Initialize video reading with PyAV (hardware-accelerated)."""
        self.container = av.open(self.input_file)
        self.stream = self.container.streams.video[0]
        
        # Try to enable VideoToolbox hardware decoding
        # This offloads H.264/HEVC decoding to Apple's media engine
        try:
            self.stream.codec_context.options = {'hwaccel': 'videotoolbox'}
        except:
            pass  # Hardware acceleration not available, continue with software
        
        # Get video properties
        self.frame_count = self.stream.frames or self._count_frames()
        self.frame_width = self.stream.width
        self.frame_height = self.stream.height
        self.fps = float(self.stream.average_rate) if self.stream.average_rate else 30.0
        
    def _count_frames(self) -> int:
        """
        Count frames manually if metadata is missing.
        
        Some video formats don't store frame count in metadata,
        so we need to iterate through the entire video to count.
        """
        count = 0
        for _ in self.container.decode(video=0):
            count += 1
        # Reset to beginning after counting
        self.container.seek(0)
        return count
    
    def _init_opencv(self):
        """Initialize video reading with OpenCV (software decoding fallback)."""
        self.cap = cv2.VideoCapture(self.input_file)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.input_file}")
        
        # Get video properties from OpenCV
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
    def read_all_frames(self, frame_step: int = 1, 
                        transform_func=None,
                        progress: bool = True) -> np.ndarray:
        """
        Read all frames from the video into memory.
        
        Args:
            frame_step: Read every Nth frame (1 = all frames, 2 = every other, etc.)
            transform_func: Optional function to apply to each frame (e.g., rotation)
            progress: Whether to show a progress bar
        
        Returns:
            Numpy array of shape (num_frames, height, width, 3) containing all frames
        """
        if self.use_hardware:
            return self._read_frames_pyav(frame_step, transform_func, progress)
        else:
            return self._read_frames_opencv(frame_step, transform_func, progress)
    
    def _read_frames_pyav(self, frame_step: int, transform_func, progress: bool) -> np.ndarray:
        """Read frames using PyAV with hardware acceleration."""
        # Reset to beginning of video
        self.container.seek(0)
        
        frames = []
        frame_idx = 0
        
        # Set up iterator with optional progress bar
        iterator = self.container.decode(video=0)
        if progress:
            iterator = tqdm(iterator, total=self.frame_count, desc="Loading (HW)")
        
        for frame in iterator:
            # Only keep frames at the specified step interval
            if frame_idx % frame_step == 0:
                # Convert PyAV frame to numpy array (RGB format)
                img = frame.to_ndarray(format='rgb24')
                
                # Apply transformation if provided
                if transform_func:
                    img = transform_func(img)
                
                frames.append(img)
            
            frame_idx += 1
        
        return np.array(frames)
    
    def _read_frames_opencv(self, frame_step: int, transform_func, progress: bool) -> np.ndarray:
        """Read frames using OpenCV (software decoding fallback)."""
        # Reset to beginning of video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frames = []
        
        # Set up iterator with optional progress bar
        iterator = range(0, self.frame_count, frame_step)
        if progress:
            iterator = tqdm(iterator, desc="Loading (SW)")
        
        for i in iterator:
            # Seek to the specific frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # OpenCV reads in BGR format, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transformation if provided
            if transform_func:
                frame = transform_func(frame)
            
            frames.append(frame)
        
        return np.array(frames)
    
    def close(self):
        """Release video resources."""
        if self.use_hardware:
            self.container.close()
        else:
            self.cap.release()


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class AppleSiliconProcessor:
    """
    Main time-based photography processor optimized for Apple Silicon.
    
    This class handles the core workflow:
        1. Load video frames into memory
        2. Optionally interpolate frames to increase count
        3. For each x-position, extract vertical slices and create panorama
        4. Optionally combine panoramas into an output video
    
    Optimizations for Apple Silicon:
        - Hardware video decoding via VideoToolbox
        - GPU array operations via MLX framework
        - ThreadPoolExecutor (takes advantage of unified memory)
        - Hardware video encoding for output
    
    Attributes:
        input_file (str): Path to input video
        output_dir (Path): Directory for output files
        slice_width (int): Width of vertical slice to extract from each frame
        frame_step (int): Process every Nth frame
        swap_dimensions (bool): Whether to swap width/height
        rotate (int): Rotation angle (0, 90, 180, 270)
        use_gpu (bool): Whether MLX GPU acceleration is enabled
        interpolate (int): Frame interpolation multiplier
        frames (np.ndarray): Loaded frames in memory
        frames_mlx: Frames as MLX array on GPU (if use_gpu=True)
    """
    
    def __init__(self, input_file: str, output_dir: str, slice_width: int = 1,
                 frame_step: int = 1, swap_dimensions: bool = False, 
                 rotate: int = 0, use_gpu: bool = True, use_hardware_decode: bool = True,
                 interpolate: int = 1, interpolate_method: str = 'auto',
                 interpolate_quality: str = 'medium'):
        """
        Initialize the processor.
        
        Args:
            input_file: Path to the input video file
            output_dir: Directory where output files will be saved
            slice_width: Width of vertical slice to extract from each frame (pixels)
            frame_step: Process every Nth frame (1 = all, 2 = half, etc.)
            swap_dimensions: Swap width/height (fixes some videos with wrong metadata)
            rotate: Rotate frames by this angle (0, 90, 180, or 270 degrees)
            use_gpu: Use MLX GPU acceleration (requires mlx package)
            use_hardware_decode: Use VideoToolbox hardware video decoding
            interpolate: Frame interpolation multiplier (1=none, 2=double, 4=quad, 8=oct)
            interpolate_method: Interpolation method ('auto', 'ffmpeg', 'rife', 'opencv')
            interpolate_quality: Interpolation quality ('fast', 'medium', 'best')
        """
        # Store configuration
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.slice_width = slice_width
        self.frame_step = frame_step
        self.swap_dimensions = swap_dimensions
        self.rotate = rotate
        self.use_gpu = use_gpu and HAS_MLX  # Only enable if MLX is available
        self.use_hardware_decode = use_hardware_decode
        self.interpolate = interpolate
        self.interpolate_quality = interpolate_quality
        
        # Resolve 'auto' interpolation method - default to FFmpeg (most reliable)
        if interpolate_method == 'auto':
            self.interpolate_method = 'ffmpeg'
        else:
            self.interpolate_method = interpolate_method
        
        # For FFmpeg interpolation, we pre-process the entire video file
        # Other methods work in-memory after loading frames
        self.temp_dir = None
        self.working_input = input_file
        
        if interpolate > 1 and self.interpolate_method == 'ffmpeg':
            self._preprocess_video_interpolation()
        
        # Open video and get properties
        self.reader = VideoReader(self.working_input, use_hardware=use_hardware_decode)
        
        # Get frame dimensions, applying transformations
        w = self.reader.frame_width
        h = self.reader.frame_height
        
        # Apply dimension swap if requested
        if swap_dimensions:
            w, h = h, w
            
        # 90° and 270° rotations swap width and height
        if rotate in (90, 270):
            w, h = h, w
        
        self.frame_width = w
        self.frame_height = h
        self.frame_count = self.reader.frame_count
        self.fps = self.reader.fps
        
        # Calculate output dimensions
        # effective_frames = how many frames we'll actually use
        self.effective_frames = self.frame_count // frame_step
        # output_width = width of each panorama image
        self.output_width = self.effective_frames * slice_width
        
        # Frame storage (will be populated by load_frames())
        self.frames: Optional[np.ndarray] = None  # CPU array
        self.frames_mlx = None  # GPU array (MLX)
        
        # Thread safety lock for parallel file writing
        self.write_lock = Lock()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print configuration summary
        print(f"Video: {self.frame_width}x{self.frame_height}, "
              f"{self.frame_count} frames @ {self.fps:.1f} fps")
        if self.interpolate > 1:
            effective_frames = self.frame_count * self.interpolate
            print(f"Interpolation: {self.interpolate}x ({self.interpolate_method}) "
                  f"→ ~{effective_frames} effective frames")
        print(f"Output: {self.output_width}x{self.frame_height} pixels per panorama")
        print(f"Backend: {'PyAV + VideoToolbox' if HAS_PYAV and use_hardware_decode else 'OpenCV'}")
        if self.use_gpu:
            print(f"GPU acceleration: MLX (single-threaded, use --no-gpu for multi-threaded CPU)")
        else:
            print(f"GPU acceleration: Disabled (multi-threaded CPU mode)")
    
    def _preprocess_video_interpolation(self):
        """
        Pre-process video with FFmpeg frame interpolation.
        
        This creates a temporary interpolated video file that will be used
        instead of the original input. The temp file is cleaned up when
        close() is called.
        """
        print(f"\nPre-processing: Interpolating frames {self.interpolate}x with FFmpeg...")
        
        # Create temporary directory for interpolated video
        self.temp_dir = tempfile.mkdtemp(prefix="tbp_interpolated_")
        interpolated_path = os.path.join(self.temp_dir, "interpolated.mp4")
        
        # Run interpolation
        interpolator = FrameInterpolator(method='ffmpeg')
        interpolator.interpolate_video_ffmpeg(
            self.input_file, 
            interpolated_path, 
            self.interpolate,
            quality=self.interpolate_quality
        )
        
        # Use interpolated video as input from now on
        self.working_input = interpolated_path
        print(f"Interpolated video saved to temp file\n")
    
    def _make_transform_func(self):
        """
        Create a frame transformation function based on current settings.
        
        Returns a function that applies swap_dimensions and rotation
        transformations to a frame.
        """
        def transform(frame):
            # Swap dimensions (transpose) if requested
            # This swaps the width and height axes
            if self.swap_dimensions:
                frame = np.transpose(frame, (1, 0, 2))
            
            # Apply rotation using numpy's rot90
            # k parameter specifies number of 90° counter-clockwise rotations
            if self.rotate == 90:
                frame = np.rot90(frame, k=3)  # 90° CW = 270° CCW = k=3
            elif self.rotate == 180:
                frame = np.rot90(frame, k=2)  # 180° = k=2
            elif self.rotate == 270:
                frame = np.rot90(frame, k=1)  # 270° CW = 90° CCW = k=1
            
            return frame
        
        return transform
    
    def load_frames(self) -> np.ndarray:
        """
        Load all video frames into memory.
        
        This reads frames from the video file (using hardware decoding if available),
        applies transformations, optionally interpolates frames, and optionally
        moves the data to GPU memory for accelerated processing.
        
        Returns:
            Numpy array of frames with shape (num_frames, height, width, 3)
        """
        print("Loading frames...")
        
        # Create transformation function for rotation/dimension swap
        transform = self._make_transform_func()
        
        # Read all frames from video
        self.frames = self.reader.read_all_frames(
            frame_step=self.frame_step,
            transform_func=transform,
            progress=True
        )
        
        print(f"Loaded {len(self.frames)} frames ({self.frames.nbytes / 1e9:.2f} GB)")
        print(f"Frame array shape: {self.frames.shape}")
        
        # Apply in-memory interpolation if using RIFE or OpenCV method
        # (FFmpeg interpolation was already done on the video file)
        if self.interpolate > 1 and self.interpolate_method in ('rife', 'opencv'):
            interpolator = FrameInterpolator(method=self.interpolate_method)
            
            if self.interpolate_method == 'rife':
                self.frames = interpolator.interpolate_frames_rife(self.frames, self.interpolate)
            else:
                self.frames = interpolator.interpolate_frames_opencv(self.frames, self.interpolate)
            
            print(f"After interpolation: {len(self.frames)} frames "
                  f"({self.frames.nbytes / 1e9:.2f} GB)")
            
            # Update output width based on new frame count
            self.effective_frames = len(self.frames)
            self.output_width = self.effective_frames * self.slice_width
        
        # Move frames to GPU if MLX acceleration is enabled
        if self.use_gpu:
            print("Moving frames to GPU (MLX)...")
            self.frames_mlx = mx.array(self.frames)
            mx.eval(self.frames_mlx)  # Force evaluation to complete transfer
            print("Frames on GPU")
        
        return self.frames
    
    def create_single_panorama_gpu(self, slice_x: int) -> np.ndarray:
        """
        Create a single panorama image using GPU acceleration (MLX).
        
        This extracts vertical slices from all frames at the specified x-position
        and concatenates them into a panoramic image. Operations run on the GPU.
        
        Args:
            slice_x: The x-coordinate (column) to extract from each frame
        
        Returns:
            Panorama image as numpy array with shape (height, output_width, 3)
        """
        end_x = min(slice_x + self.slice_width, self.frame_width)
        
        # Extract slices from all frames on GPU
        # Shape: (num_frames, height, slice_width, 3)
        slices = self.frames_mlx[:, :, slice_x:end_x, :]
        
        # Transpose to (height, num_frames, slice_width, 3)
        # This puts the frame dimension second so reshape works correctly
        slices = mx.transpose(slices, (1, 0, 2, 3))
        
        # Reshape to (height, num_frames * slice_width, 3)
        # This concatenates all slices horizontally
        panorama = mx.reshape(slices, (self.frame_height, -1, 3))
        
        # Transfer back to CPU and convert to uint8 numpy array
        mx.eval(panorama)
        return np.array(panorama, dtype=np.uint8)
    
    def create_single_panorama_cpu(self, slice_x: int) -> np.ndarray:
        """
        Create a single panorama image using CPU (NumPy).
        
        This is the CPU-based version of create_single_panorama_gpu().
        Used when GPU acceleration is disabled or unavailable.
        
        Args:
            slice_x: The x-coordinate (column) to extract from each frame
        
        Returns:
            Panorama image as numpy array with shape (height, output_width, 3)
        """
        end_x = min(slice_x + self.slice_width, self.frame_width)
        
        # Extract slices from all frames
        # Shape: (num_frames, height, slice_width, 3)
        slices = self.frames[:, :, slice_x:end_x, :]
        
        # Transpose to (height, num_frames, slice_width, 3)
        slices = np.transpose(slices, (1, 0, 2, 3))
        
        # Reshape to concatenate slices horizontally
        panorama = slices.reshape(self.frame_height, -1, 3)
        
        return panorama
    
    def create_single_panorama(self, slice_x: int) -> np.ndarray:
        """
        Create a single panorama image using the best available method.
        
        Automatically selects GPU or CPU based on configuration.
        
        Args:
            slice_x: The x-coordinate (column) to extract from each frame
        
        Returns:
            Panorama image as numpy array
        """
        if self.use_gpu and self.frames_mlx is not None:
            return self.create_single_panorama_gpu(slice_x)
        else:
            return self.create_single_panorama_cpu(slice_x)
    
    def _save_panorama(self, slice_x: int) -> str:
        """
        Create and save a single panorama (thread-safe).
        
        This is a wrapper for use with ThreadPoolExecutor that handles
        both creating the panorama and saving it to disk.
        
        Args:
            slice_x: The x-coordinate for this panorama
        
        Returns:
            Path to the saved panorama image
        """
        panorama = self.create_single_panorama(slice_x)
        output_path = self.output_dir / f"pan_img-{slice_x:04d}.jpg"
        
        # Use lock to ensure thread-safe file writing
        with self.write_lock:
            Image.fromarray(panorama).save(output_path, quality=95)
        
        return str(output_path)
    
    def create_all_panoramas(self, start_x: int = 0, end_x: Optional[int] = None,
                             step_x: int = 1, num_threads: int = 4) -> List[str]:
        """
        Create panorama images for all (or a range of) x-positions.
        
        This is the main processing method. It creates one panorama image
        for each x-position in the specified range.
        
        Args:
            start_x: Starting x-position (default: 0)
            end_x: Ending x-position (default: frame_width)
            step_x: Step between x-positions (default: 1)
            num_threads: Number of threads for parallel processing
        
        Returns:
            List of paths to the created panorama images
        """
        if end_x is None:
            end_x = self.frame_width
        
        # MLX/Metal is NOT thread-safe - must use single thread if using GPU
        if self.use_gpu and num_threads > 1:
            print(f"NOTE: MLX GPU mode requires single-threaded execution.")
            print(f"      Using 1 thread instead of {num_threads}.")
            print(f"      (Use --no-gpu for multi-threaded CPU mode)")
            num_threads = 1
        
        # Load all frames into memory
        self.load_frames()
        
        x_positions = list(range(start_x, end_x, step_x))
        print(f"\nGenerating {len(x_positions)} panoramas using {num_threads} thread(s)...")
        
        output_files = []
        
        if num_threads > 1:
            # Parallel processing with ThreadPoolExecutor
            # ThreadPoolExecutor is better than ProcessPoolExecutor on Apple Silicon
            # because unified memory means no data copying between threads
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self._save_panorama, x): x for x in x_positions}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Panoramas"):
                    output_files.append(future.result())
        else:
            # Sequential processing (required for GPU mode)
            for x in tqdm(x_positions, desc="Panoramas"):
                output_files.append(self._save_panorama(x))
        
        return sorted(output_files)
    
    def create_video_from_panoramas(self, output_name: str, fps: int = 30) -> str:
        """
        Create a video from all panorama images in the output directory.
        
        This combines all the panorama images into a video file, which shows
        how the perspective shifts as we move the slice position across the frame.
        
        Args:
            output_name: Name for the output video (without extension)
            fps: Frame rate for the output video
        
        Returns:
            Path to the created video file
        """
        print("\nCreating video from panoramas...")
        
        # Find all panorama images
        images = sorted(self.output_dir.glob("pan_img-*.jpg"))
        if not images:
            raise ValueError("No panorama images found")
        
        output_path = self.output_dir / f"{output_name}.mp4"
        
        if HAS_PYAV:
            # Use PyAV for video writing (can use hardware encoding)
            first_img = np.array(Image.open(images[0]))
            height, width = first_img.shape[:2]
            
            # H.264 requires even dimensions due to YUV 4:2:0 chroma subsampling
            # Pad by 1 pixel if dimensions are odd
            pad_width = width % 2
            pad_height = height % 2
            if pad_width or pad_height:
                print(f"Padding dimensions from {width}x{height} to "
                      f"{width + pad_width}x{height + pad_height} (H.264 requires even dimensions)")
            
            output_width = width + pad_width
            output_height = height + pad_height
            
            # Create video container and stream
            container = av.open(str(output_path), mode='w')
            stream = container.add_stream('h264', rate=fps)
            stream.width = output_width
            stream.height = output_height
            stream.pix_fmt = 'yuv420p'  # Standard pixel format for H.264
            
            # Try to use VideoToolbox hardware encoder
            try:
                stream.options = {'c:v': 'h264_videotoolbox'}
            except:
                pass  # Fall back to software encoding
            
            # Write each panorama as a video frame
            for img_path in tqdm(images, desc="Writing video"):
                img = np.array(Image.open(img_path))
                
                # Pad if necessary (using edge pixels)
                if pad_width or pad_height:
                    img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='edge')
                
                # Convert to PyAV frame and encode
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            
            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
            
            container.close()
        else:
            # Fallback to OpenCV for video writing
            import cv2
            first_img = cv2.imread(str(images[0]))
            height, width = first_img.shape[:2]
            
            # Handle odd dimensions
            pad_width = width % 2
            pad_height = height % 2
            output_width = width + pad_width
            output_height = height + pad_height
            
            if pad_width or pad_height:
                print(f"Padding dimensions from {width}x{height} to {output_width}x{output_height}")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
            
            for img_path in tqdm(images, desc="Writing video"):
                img = cv2.imread(str(img_path))
                if pad_width or pad_height:
                    img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_REPLICATE)
                video.write(img)
            
            video.release()
        
        print(f"Video saved to: {output_path}")
        return str(output_path)
    
    def close(self):
        """
        Release all resources and clean up.
        
        This should be called when done processing to:
        - Release video file handles
        - Free frame memory (CPU and GPU)
        - Delete temporary files from FFmpeg interpolation
        """
        # Release video reader
        self.reader.close()
        
        # Free CPU frame memory
        if self.frames is not None:
            del self.frames
            self.frames = None
        
        # Free GPU frame memory
        if self.frames_mlx is not None:
            del self.frames_mlx
            self.frames_mlx = None
        
        # Clean up temporary directory from FFmpeg interpolation
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
        
        # Force garbage collection to free memory immediately
        gc.collect()
        print("Memory released.")


# =============================================================================
# BENCHMARK UTILITY
# =============================================================================

def benchmark_backends(input_file: str, num_frames: int = 100):
    """
    Benchmark different video decoding backends.
    
    This helps users determine which backend is fastest on their system.
    
    Args:
        input_file: Path to a video file to use for benchmarking
        num_frames: Number of frames to read in benchmark
    """
    print("=" * 60)
    print("BENCHMARKING VIDEO DECODING BACKENDS")
    print("=" * 60)
    
    results = {}
    
    # Test PyAV with hardware decoding
    if HAS_PYAV:
        print("\n[1] PyAV + VideoToolbox (hardware)...")
        try:
            reader = VideoReader(input_file, use_hardware=True)
            start = time.time()
            frames = reader.read_all_frames(frame_step=1, progress=False)
            frames = frames[:num_frames] if len(frames) > num_frames else frames
            elapsed = time.time() - start
            reader.close()
            fps = len(frames) / elapsed
            results['PyAV (HW)'] = fps
            print(f"   → {fps:.1f} frames/sec")
            del frames
        except Exception as e:
            print(f"   → Failed: {e}")
    
    # Test OpenCV (software decoding)
    print("\n[2] OpenCV (software)...")
    try:
        import cv2
        reader = VideoReader(input_file, use_hardware=False)
        start = time.time()
        frames = reader.read_all_frames(frame_step=1, progress=False)
        frames = frames[:num_frames] if len(frames) > num_frames else frames
        elapsed = time.time() - start
        reader.close()
        fps = len(frames) / elapsed
        results['OpenCV (SW)'] = fps
        print(f"   → {fps:.1f} frames/sec")
        del frames
    except Exception as e:
        print(f"   → Failed: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    for backend, fps in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {backend}: {fps:.1f} frames/sec")
    
    if results:
        best = max(results.items(), key=lambda x: x[1])
        print(f"\nRecommendation: Use {best[0]}")
    
    gc.collect()
    return results


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """
    Main entry point for command-line usage.
    
    Parses command-line arguments and runs the appropriate processing.
    """
    # Create argument parser with detailed help
    parser = argparse.ArgumentParser(
        prog='tbp_optimized.py',
        description="""
Time-Based Photography Tool (Apple Silicon Optimized)

Creates panoramic images by extracting vertical slices from each frame
of a video and concatenating them. The result compresses an entire
video's timeline into a single static image.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Basic usage (GPU mode, single-threaded):
    python tbp_optimized.py video.mp4 output/

  Multi-threaded CPU mode (often faster for many panoramas):
    python tbp_optimized.py video.mp4 output/ --no-gpu --threads 8

  Fix scrambled output (dimension issues):
    python tbp_optimized.py video.mp4 output/ --swap-dimensions
    python tbp_optimized.py video.mp4 output/ --rotate 90

  Frame interpolation for low-fps videos (wider output):
    python tbp_optimized.py video.mp4 output/ --interpolate 2
    python tbp_optimized.py video.mp4 output/ --interpolate 4 --interpolate-quality fast  # ~5x faster

  Create video and clean up images:
    python tbp_optimized.py video.mp4 output/ --make-video --cleanup

  Benchmark to find fastest backend:
    python tbp_optimized.py video.mp4 output/ --benchmark

INTERPOLATION METHODS:
  auto    : Uses FFmpeg (most reliable, default)
  ffmpeg  : FFmpeg minterpolate filter (recommended)
            Quality presets via --interpolate-quality:
              fast   : Blend mode (~5x faster, lower quality)
              medium : MCI/OBMC (~2x faster, good quality) [default]
              best   : MCI/AOBMC (slowest, highest quality)
  rife    : RIFE neural network (best quality, needs: pip install rife-ncnn-vulkan-python-tntwise)
  opencv  : OpenCV optical flow (basic quality, no extra deps)

PERFORMANCE TIPS:
  - Use --interpolate-quality fast for ~5x faster frame interpolation
  - GPU mode (default): Fast per-panorama, but single-threaded
  - CPU mode (--no-gpu --threads 8): Parallel processing, often faster overall
  - For 1000+ panoramas, try both modes to see which is faster on your hardware
  - Hardware acceleration: VideoToolbox used for decode/encode on Apple Silicon

INSTALLATION:
  Required: pip install numpy pillow tqdm av
  Optional: pip install mlx rife-ncnn-vulkan-python-tntwise
  Also needs: brew install ffmpeg (for some interpolation features)

For more information, see the docstring at the top of this script.
        """
    )
    
    # ===================
    # Positional arguments
    # ===================
    parser.add_argument(
        "input",
        metavar="INPUT_VIDEO",
        help="Path to the input video file (e.g., video.mp4, clip.mov)"
    )
    parser.add_argument(
        "output_dir",
        metavar="OUTPUT_DIR",
        help="Directory where output files will be saved"
    )
    
    # =======================
    # Slice/frame parameters
    # =======================
    slice_group = parser.add_argument_group('Slice Parameters')
    slice_group.add_argument(
        "--slice-width", 
        type=int, 
        default=1,
        metavar="N",
        help="Width of each vertical slice in pixels (default: 1)"
    )
    slice_group.add_argument(
        "--frame-step",
        type=int, 
        default=1,
        metavar="N",
        help="Process every Nth frame (default: 1 = all frames)"
    )
    slice_group.add_argument(
        "--start-x",
        type=int,
        default=0,
        metavar="N",
        help="Starting x-position for panorama generation (default: 0)"
    )
    slice_group.add_argument(
        "--end-x",
        type=int,
        default=None,
        metavar="N",
        help="Ending x-position for panorama generation (default: frame width)"
    )
    slice_group.add_argument(
        "--step-x",
        type=int,
        default=1,
        metavar="N",
        help="Step between x-positions (default: 1)"
    )
    
    # =======================
    # Transform parameters
    # =======================
    transform_group = parser.add_argument_group('Transform Options')
    transform_group.add_argument(
        "--swap-dimensions",
        action="store_true",
        help="Swap width and height (fixes some videos with incorrect metadata)"
    )
    transform_group.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        metavar="DEG",
        help="Rotate frames clockwise by degrees: 0, 90, 180, or 270 (default: 0)"
    )
    
    # =======================
    # Interpolation parameters
    # =======================
    interp_group = parser.add_argument_group('Frame Interpolation')
    interp_group.add_argument(
        "--interpolate",
        type=int,
        default=1,
        choices=[1, 2, 4, 8],
        metavar="N",
        help="Frame interpolation multiplier: 1 (none), 2, 4, or 8 (default: 1)"
    )
    interp_group.add_argument(
        "--interpolate-method",
        type=str,
        default='auto',
        choices=['auto', 'ffmpeg', 'rife', 'opencv'],
        metavar="METHOD",
        help="Interpolation method: auto, ffmpeg, rife, or opencv (default: auto)"
    )
    interp_group.add_argument(
        "--interpolate-quality",
        type=str,
        default='medium',
        choices=['fast', 'medium', 'best'],
        metavar="QUALITY",
        help="Interpolation quality: fast (~5x faster), medium (default), best (slowest)"
    )
    
    # =======================
    # Output parameters
    # =======================
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "--make-video",
        action="store_true",
        help="Create a video from the generated panorama images"
    )
    output_group.add_argument(
        "--video-fps",
        type=int,
        default=30,
        metavar="N",
        help="Frame rate for output video (default: 30)"
    )
    output_group.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete panorama images after creating video (use with --make-video)"
    )
    
    # =======================
    # Performance parameters
    # =======================
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument(
        "--threads",
        type=int,
        default=4,
        metavar="N",
        help="Number of threads for CPU mode (default: 4, ignored in GPU mode)"
    )
    perf_group.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable MLX GPU acceleration (enables multi-threading)"
    )
    perf_group.add_argument(
        "--no-hardware-decode",
        action="store_true",
        help="Disable VideoToolbox hardware video decoding"
    )
    perf_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark video decoding backends and exit"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # =====================
    # Print system info
    # =====================
    print("=" * 60)
    print("TIME-BASED PHOTOGRAPHY - APPLE SILICON OPTIMIZED")
    print("=" * 60)
    print(f"PyAV (VideoToolbox): {'Available' if HAS_PYAV else 'Not installed (pip install av)'}")
    print(f"MLX (GPU):           {'Available' if HAS_MLX else 'Not installed (pip install mlx)'}")
    
    # Show RIFE status with more detail
    if HAS_RIFE:
        print(f"RIFE (interpolation): Installed (use --interpolate-method rife to try it)")
    elif RIFE_ERROR:
        print(f"RIFE (interpolation): Error - {RIFE_ERROR}")
    else:
        print(f"RIFE (interpolation): Not installed (pip install rife-ncnn-vulkan-python-tntwise)")
    print(f"Default interpolation: FFmpeg minterpolate (most reliable)")
    print()
    
    # Handle benchmark mode
    if args.benchmark:
        benchmark_backends(args.input)
        return
    
    # Warn if --cleanup used without --make-video
    if args.cleanup and not args.make_video:
        print("WARNING: --cleanup has no effect without --make-video")
    
    # Start timing
    start_time = time.time()
    
    # Create processor
    processor = AppleSiliconProcessor(
        args.input, 
        args.output_dir,
        slice_width=args.slice_width,
        frame_step=args.frame_step,
        swap_dimensions=args.swap_dimensions,
        rotate=args.rotate,
        use_gpu=not args.no_gpu,
        use_hardware_decode=not args.no_hardware_decode,
        interpolate=args.interpolate,
        interpolate_method=args.interpolate_method,
        interpolate_quality=args.interpolate_quality
    )
    
    # Generate panoramas
    output_files = processor.create_all_panoramas(
        start_x=args.start_x,
        end_x=args.end_x,
        step_x=args.step_x,
        num_threads=args.threads
    )
    
    # Optionally create video
    if args.make_video:
        project_name = Path(args.input).stem
        processor.create_video_from_panoramas(project_name, fps=args.video_fps)
        
        # Optionally clean up panorama images
        if args.cleanup:
            print("Cleaning up panorama images...")
            count = 0
            for img_path in Path(args.output_dir).glob("pan_img-*.jpg"):
                img_path.unlink()
                count += 1
            print(f"Deleted {count} images.")
    
    # Clean up
    processor.close()
    
    # Print elapsed time
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
